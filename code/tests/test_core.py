"""CPU-only checks on small artificial tensors; these are not experiment data."""
import math
import unittest

import torch

from g2p_wam.diagnosis import confidence_score, outcome_auc, score_motion_correlation
from g2p_wam.geodpo import StreamPredictions, geodpo_loss
from g2p_wam.geosft import StaticReadout, DynamicReadout, dynamic_loss, weighted_average
from g2p_wam.preferences import EpisodeScore, build_pairs
from g2p_wam.teachers import FeatureTargets, TrackTargets


def stream(winner=0.0, loser=1.0, reference=0.5, mask=None):
    def tensor(value):
        return torch.full((2, 3), value, dtype=torch.float32, requires_grad=True)
    return StreamPredictions(tensor(winner), tensor(loser), tensor(reference),
                             tensor(reference), tensor(0), tensor(0), mask)


class CoreTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(13)
        torch.set_num_threads(1)

    def test_confidence_mean_and_detach(self):
        values = torch.arange(48.0).reshape(2, 2, 3, 4).requires_grad_()
        result = confidence_score(values)
        torch.testing.assert_close(result, torch.tensor([11.5, 35.5]))
        self.assertFalse(result.requires_grad)

    def test_confidence_bad_input(self):
        with self.assertRaises(ValueError):
            confidence_score(torch.ones(2, 3))
        with self.assertRaises(ValueError):
            confidence_score(torch.full((1, 1, 1, 1), float("nan")))

    def test_auc_ties_and_order(self):
        self.assertEqual(outcome_auc([1, 2, 2, 3], [0, 0, 1, 1]), 0.875)
        self.assertEqual(outcome_auc([4, 4], [0, 1]), 0.5)
        self.assertEqual(outcome_auc([2, 1], [1, 0]), 1.0)
        self.assertEqual(outcome_auc([2, 1], [0, 1]), 0.0)

    def test_auc_matches_pairwise(self):
        scores = torch.randint(0, 4, (19,)).float()
        y = torch.tensor([0, 1] * 9 + [0])
        pos, neg = scores[y == 1], scores[y == 0]
        expected = ((pos[:, None] > neg).float() + 0.5 * (pos[:, None] == neg)).mean().item()
        self.assertAlmostEqual(outcome_auc(scores, y), expected, places=6)

    def test_python_float_precision_is_preserved(self):
        self.assertEqual(outcome_auc([1.0, 1.0 + 1e-8], [0, 1]), 1.0)
        rho = score_motion_correlation([1.0, 1.0 + 1e-8], [2.0, 2.0 + 1e-8])
        self.assertAlmostEqual(rho, 1.0, places=12)

    def test_half_precision_weighted_reduction(self):
        values = torch.tensor([60000., 60000.], dtype=torch.float16, requires_grad=True)
        result = weighted_average(values, torch.ones_like(values))
        self.assertEqual(result.item(), 60000.)
        result.backward()
        torch.testing.assert_close(values.grad, torch.full_like(values, 0.5))

    def test_auc_undefined_not_zero(self):
        self.assertTrue(math.isnan(outcome_auc([], [])))
        self.assertTrue(math.isnan(outcome_auc([1, 2], [1, 1])))
        with self.assertRaises(ValueError):
            outcome_auc([1, 2], [0, 2])

    def test_motion_correlation(self):
        self.assertAlmostEqual(score_motion_correlation([1, 2, 3], [6, 4, 2]), -1.0)
        self.assertTrue(math.isnan(score_motion_correlation([1, 1], [2, 3])))
        with self.assertRaises(ValueError):
            score_motion_correlation([1], [2, 3])

    def test_static_loss_and_student_gradients(self):
        readout = StaticReadout(4, 6, 8)
        student = torch.randn(2, 3, 4, requires_grad=True)
        teacher = torch.randn(2, 3, 6, requires_grad=True)
        weights = torch.rand(2, 3, requires_grad=True)
        output = readout(student, FeatureTargets(teacher, weights), scale_weight=0.3)
        torch.testing.assert_close(output["loss"], output["angular"] + 0.3 * output["scale"])
        output["loss"].backward()
        self.assertGreater(student.grad.abs().sum().item(), 0)
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in readout.parameters()))
        self.assertIsNone(teacher.grad)
        self.assertIsNone(weights.grad)

    def test_static_empty_mask(self):
        readout = StaticReadout(3, 4, 6)
        student = torch.randn(2, 3, requires_grad=True)
        loss = readout(student, FeatureTargets(torch.randn(2, 4), torch.zeros(2)))["loss"]
        self.assertEqual(loss.item(), 0)
        loss.backward()
        torch.testing.assert_close(student.grad, torch.zeros_like(student))

    def test_feature_contract(self):
        with self.assertRaises(ValueError):
            FeatureTargets(torch.ones(2, 3), torch.ones(3)).detached()
        with self.assertRaises(ValueError):
            FeatureTargets(torch.ones(2, 3), -torch.ones(2)).detached()
        with self.assertRaises(ValueError):
            StaticReadout(2, 3, 4)(torch.ones(3, 2), FeatureTargets(torch.ones(4, 3)))

    def test_trajectory_increments_and_visibility(self):
        xyz = torch.tensor([[[[0., 0, 0], [1, 2, 3], [3, 5, 7]]]], requires_grad=True)
        target = TrackTargets(xyz, torch.tensor([[[1., 1, 0]]]))
        flow, mask = target.increments()
        torch.testing.assert_close(flow, torch.tensor([[[[1., 2, 3], [2, 3, 4]]]]))
        torch.testing.assert_close(mask, torch.tensor([[[1., 0]]]))
        self.assertFalse(flow.requires_grad)

    def test_trajectory_confidence_filter(self):
        xyz = torch.zeros(1, 2, 3, 3)
        confidence = torch.tensor([[[1., 1, 1], [4, 4, 4]]], requires_grad=True)
        flow, weights = TrackTargets(xyz, torch.ones(1, 2, 3), confidence).increments(confidence_filter=True)
        self.assertEqual(flow.shape, (1, 2, 2, 3))
        self.assertFalse(weights.requires_grad)
        # Median includes ties, rather than forcing an exact half-count.
        self.assertEqual(weights.sum().item(), 4)
        with self.assertRaises(ValueError):
            TrackTargets(xyz, torch.ones(1, 2, 3)).increments(confidence_filter=True)

    def test_dynamic_masked_loss(self):
        xyz = torch.zeros(1, 1, 3, 3, requires_grad=True)
        predicted = torch.ones(1, 1, 2, 3, requires_grad=True)
        target = TrackTargets(xyz, torch.tensor([[[1., 1, 0]]]))
        loss = dynamic_loss(predicted, target, beta=0.1)
        self.assertAlmostEqual(loss.item(), 2.85, places=5)
        loss.backward()
        torch.testing.assert_close(predicted.grad[0, 0, 1], torch.zeros(3))
        self.assertIsNone(xyz.grad)

    def test_dynamic_empty_mask(self):
        prediction = torch.randn(1, 2, 2, 3, requires_grad=True)
        loss = dynamic_loss(prediction, TrackTargets(torch.randn(1, 2, 3, 3), torch.zeros(1, 2, 3)))
        self.assertEqual(loss.item(), 0)
        loss.backward()
        torch.testing.assert_close(prediction.grad, torch.zeros_like(prediction))

    def test_dynamic_readout_shape_and_gradients(self):
        model = DynamicReadout(5, width=8, heads=2, layers=1, bands=2, max_steps=4)
        video = torch.randn(2, 4, 5, requires_grad=True)
        uv = torch.rand(3, 2, requires_grad=True)
        times = torch.tensor([0, 1])
        output = model(video, uv, times)
        self.assertEqual(output.shape, (2, 3, 2, 3))
        torch.testing.assert_close(output, torch.zeros_like(output))
        # After the zero-initialized final layer learns, gradients reach video states.
        with torch.no_grad():
            model.output[-1].weight.normal_(std=0.1)
        model(video, uv, times).square().sum().backward()
        self.assertGreater(video.grad.abs().sum().item(), 0)
        self.assertIsNone(uv.grad)
        with self.assertRaises(ValueError):
            model(video, uv, torch.tensor([4]))

    def test_rank_filter_count_and_reproducibility(self):
        pool = [EpisodeScore(f"s{k}", "task", True, float(k)) for k in range(5)]
        pool += [EpisodeScore(f"f{k}", "task", False, float(k)) for k in range(3)]
        result = build_pairs(pool, pairs_per_group=40, seed=1)
        self.assertEqual(len(result), 40)
        self.assertEqual({r.winner_id for r in result}, {"s3", "s4"})
        self.assertEqual({r.loser_id for r in result}, {"f0"})
        self.assertEqual(result, build_pairs(list(reversed(pool)), pairs_per_group=40, seed=1))

    def test_pairing_does_not_enforce_margin(self):
        pool = [EpisodeScore("s", "g", True, 1), EpisodeScore("f", "g", False, 9)]
        pair = build_pairs(pool, pairs_per_group=1)[0]
        self.assertEqual((pair.winner_id, pair.loser_id), ("s", "f"))

    def test_missing_outcome_group_is_skipped(self):
        self.assertEqual(build_pairs([], pairs_per_group=10), [])
        self.assertEqual(build_pairs([EpisodeScore("s", "g", True, 1)], pairs_per_group=10), [])
        with self.assertRaises(ValueError):
            build_pairs([EpisodeScore("s", "g", True, 1)] * 2, pairs_per_group=2)
        with self.assertRaises(ValueError):
            build_pairs([], pairs_per_group=2, keep_fraction=0)

    def test_pairs_remain_in_groups(self):
        pool = [EpisodeScore(f"{g}{o}", g, bool(o), float(o)) for g in ["a", "b"] for o in [0, 1]]
        pairs = build_pairs(pool, pairs_per_group=5)
        self.assertEqual(len(pairs), 10)
        self.assertTrue(all(p.winner_id.startswith(p.group_id) and p.loser_id.startswith(p.group_id) for p in pairs))

    def test_equal_policy_reference_loss(self):
        output = geodpo_loss(stream(0.5, 0.5), stream(0.5, 0.5), beta=2, anchor_weight=0.2)
        self.assertAlmostEqual(output.loss.item(), 2 * math.log(2), places=6)
        self.assertEqual(output.reference_penalty.item(), 0)

    def test_preference_direction_and_anchor(self):
        good = geodpo_loss(stream(), stream(), beta=2, anchor_weight=0.2)
        bad = geodpo_loss(stream(1, 0), stream(1, 0), beta=2)
        expected_pref = 2 * math.log1p(math.exp(-2))
        self.assertAlmostEqual(good.preference.item(), expected_pref, places=6)
        self.assertAlmostEqual(good.reference_penalty.item(), 1.0, places=6)
        self.assertAlmostEqual(good.loss.item(), expected_pref + 0.2, places=6)
        self.assertLess(good.preference.item(), bad.preference.item())

    def test_geodpo_reference_and_target_stop_gradient(self):
        video, action = stream(), stream()
        geodpo_loss(video, action, beta=2, anchor_weight=0.2).loss.backward()
        for value in (video, action):
            self.assertIsNotNone(value.winner.grad)
            self.assertIsNotNone(value.loser.grad)
            self.assertGreater(value.loser.grad.abs().sum().item(), 0)
            for detached in (value.reference_winner, value.reference_loser, value.target_winner, value.target_loser):
                self.assertIsNone(detached.grad)

    def test_action_mask_ignores_invalid_channels(self):
        mask = torch.tensor([[1., 0, 0]]).expand(2, -1)
        action = stream(mask=mask)
        value = geodpo_loss(stream(), action, beta=2).loss.item()
        with torch.no_grad():
            action.winner[:, 1:] = 999
            action.loser[:, 1:] = -999
        output = geodpo_loss(stream(), action, beta=2)
        self.assertEqual(output.loss.item(), value)
        output.loss.backward()
        torch.testing.assert_close(action.winner.grad[:, 1:], torch.zeros(2, 2))

    def test_geodpo_zero_mask_and_stream_weight(self):
        video, action = stream(), stream(mask=torch.zeros(2, 3))
        result = geodpo_loss(video, action, beta=2, video_weight=0, anchor_weight=0)
        self.assertAlmostEqual(result.loss.item(), math.log(2), places=6)
        result.loss.backward()
        torch.testing.assert_close(action.loser.grad, torch.zeros_like(action.loser))
        torch.testing.assert_close(video.loser.grad, torch.zeros_like(video.loser))
        with self.assertRaises(ValueError):
            geodpo_loss(video, action, beta=0)


if __name__ == "__main__":
    unittest.main()

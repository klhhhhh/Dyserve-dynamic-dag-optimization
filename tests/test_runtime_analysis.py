import unittest

from runtime_analysis.confidence_gating import evaluate
from runtime_analysis.convert_openhands import convert_row
from runtime_analysis.replay import replay
from runtime_analysis.schema import make_event, validate_dataset
from runtime_analysis.solver_adapter import MockCapacitySolver


class RuntimeAnalysisTest(unittest.TestCase):
    def test_schema_and_validation(self):
        events = [make_event(
            workflow_id="w1", event_id="e1", sequence=0,
            event_type="node_created", timestamp_ms=1.0, node_id="n1",
            semantic_type="inspect",
        )]
        report = validate_dataset(events)
        self.assertEqual(report["valid_event_fraction"], 1.0)
        self.assertTrue(report["schema_valid"])
        self.assertFalse(report["runtime_ground_truth_ready"])

    def test_openhands_conversion_preserves_usage(self):
        row = {
            "instance_id": "task-1",
            "history": [{
                "type": "llm_call", "model": "claude",
                "usage": {"input_tokens": 100, "output_tokens": 20},
                "timestamp": "2026-01-01T00:00:00Z",
            }],
        }
        event = convert_row(row)[0]
        self.assertEqual(event["workflow_id"], "task-1")
        self.assertEqual(event["input_tokens"], 100)
        self.assertEqual(event["output_tokens"], 20)
        self.assertIsNotNone(event["timestamp_ms"])

    def test_confidence_gate_rejects_uncertain_rows(self):
        rows = [{
            "true_topology": "branch",
            "topology_ranking": ["branch", "join"],
            "topology_probabilities": {"branch": 0.9, "join": 0.1},
            "true_counts": {"reason": 1},
            "predicted_counts": {"reason": 1},
            "presence_scores": {"reason": 0.9},
        }, {
            "true_topology": "join",
            "topology_ranking": ["join", "branch"],
            "topology_probabilities": {"join": 0.51, "branch": 0.49},
            "true_counts": {"reason": 1},
            "predicted_counts": {"reason": 0},
            "presence_scores": {"reason": 0.5},
        }]
        result = evaluate(rows, 0.8, 0.5, 0.25, 1)
        self.assertEqual(result["triggered"], 1)
        self.assertEqual(result["decision_proxy_precision"], 1.0)

    def test_mock_replay_is_explicitly_non_scientific(self):
        events = [make_event(
            workflow_id="w1", event_id=f"e{i}", sequence=i,
            event_type="node_created", node_id=f"n{i}",
            semantic_type="reason" if i == 1 else "inspect",
        ) for i in range(3)]
        solver = MockCapacitySolver()
        positions = replay(events, solver, horizon=2)
        self.assertEqual(len(positions), 3)
        self.assertIn("not_dyserve_ilp", solver.scientific_validity)


if __name__ == "__main__":
    unittest.main()

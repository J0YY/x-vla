"""Tests execute under the same reference guard as the CLI."""
import unittest
import numpy as np
from research.odt_reference.run_tests import install_guards, COUNTS
from research.odt_reference.shared_dag import Graph, Node, canonicalize, common_bases, apply_bases
from research.odt_reference.curve import physical_variant
from research.odt_reference import run_curve as original
from research.odt_ffn_native_v1.reference import source_audit, anchored_basis, rank_plan
from research.odt_ffn_native_v1.common import conditions, comparison, dev_selection


def fixture():
    nodes=[Node("input",np.eye(4),source="token0")]
    for i in range(1,6):nodes.append(Node(str(i),np.eye(4),(i-1,)))
    return canonicalize(Graph(nodes,np.eye(4)))


class ReferenceTests(unittest.TestCase):
    def test_exact17_variants_and_only_ffn_rank_changes(self):
        declared=conditions()
        self.assertEqual(len(declared),17)
        self.assertEqual(len({x["name"] for x in declared}),17)
        graph=fixture();plan=rank_plan(graph,2)
        self.assertEqual(plan.ranks,(4,4,4,4,2,4))
        self.assertEqual(plan.removed_dimensions,2)
    def test_anchored_seed_reconstruction_and_physical_zero(self):
        graph=fixture();bases=common_bases(graph)
        controls,receipt=anchored_basis(graph,bases,0)
        self.assertLess(receipt["anchor_error"],1e-12)
        for i in (0,1,2,3,5):np.testing.assert_array_equal(controls[i],bases[i])
        plan=rank_plan(graph,2);variant=physical_variant(graph,controls,plan)
        raw=np.zeros((1,1,3));zero=original.homogeneous(raw)
        source=original.checked_chart(graph,zero);cut=original.checked_chart(variant.graph,zero)
        original.close(cut.decoded,source.decoded,1e-10,"zero replay")
        masked=original.checked_chart(apply_bases(graph,controls),zero,plan.ranks)
        original.close(cut.projective,masked.projective,1e-10,"mask replay")
    def test_frozen_devselection_and_strict_native_double_gates(self):
        array={"dev_image_task_ids":np.repeat(np.arange(10),4)}
        selected=dev_selection(array)
        self.assertEqual(len(selected),30)
        self.assertEqual(selected[3],{"image_index":4,"token_index":0})
        expected=np.ones((2,3))
        self.assertTrue(comparison(expected,expected,double=True)["passed"])
        self.assertFalse(comparison(expected+3e-5,expected)["passed"])
        self.assertFalse(comparison(expected+1e-8,expected,double=True)["passed"])
    def test_audit(self):
        self.assertEqual(len(source_audit()["bridge_reference"]),4)
        self.assertEqual(COUNTS["prohibited_attempts"],0)


if __name__=="__main__":
    source_audit();install_guards();unittest.main()

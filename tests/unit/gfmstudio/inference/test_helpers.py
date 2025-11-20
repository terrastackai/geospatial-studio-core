# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


import pytest

from gfmstudio.inference.v2.helpers import merge_bounding_boxes


class TestMergeBoundingBoxes:

    @pytest.mark.parametrize(
        "bbox1,bbox2,expected",
        [
            # Basic cases
            pytest.param(
                [0, 0, 10, 10], [20, 20, 30, 30], [0, 0, 30, 30], id="non_overlapping"
            ),
            pytest.param(
                [0, 0, 15, 15], [10, 10, 25, 25], [0, 0, 25, 25], id="overlapping"
            ),
            pytest.param(
                [0, 0, 20, 20], [5, 5, 15, 15], [0, 0, 20, 20], id="one_inside_another"
            ),
            pytest.param(
                [10, 10, 20, 20], [10, 10, 20, 20], [10, 10, 20, 20], id="identical"
            ),
            # Coordinate variations
            pytest.param(
                [-10, -10, -5, -5],
                [-15, -15, -8, -8],
                [-15, -15, -5, -5],
                id="negative_coords",
            ),
            pytest.param(
                [
                    36.620657549132645,
                    -1.3861564035463123,
                    36.821026102010684,
                    -1.2931153584817277,
                ],
                [36.7, -1.5, 36.9, -1.1],
                [36.620657549132645, -1.5, 36.9, -1.1],
                id="floating_point",
            ),
            # Edge cases
            pytest.param(
                [0, 0, 10, 10], [10, 0, 20, 10], [0, 0, 20, 10], id="touching_edges"
            ),
            pytest.param(
                [5, 5, 5, 5], [10, 10, 10, 10], [5, 5, 10, 10], id="single_points"
            ),
            pytest.param(
                [0, 5, 10, 5], [15, 5, 25, 5], [0, 5, 25, 5], id="horizontal_lines"
            ),
            pytest.param(
                [5, 0, 5, 10], [5, 15, 5, 25], [5, 0, 5, 25], id="vertical_lines"
            ),
        ],
    )
    def test_merge_bounding_boxes(self, bbox1, bbox2, expected):
        """Test all merge scenarios using pytest parametrize"""
        result = merge_bounding_boxes(bbox1, bbox2)
        assert result == expected

import unittest

from partground.geometry import ViewBox, intersect_views


class GeometryTests(unittest.TestCase):
    def test_orthographic_intersection_in_world_coordinates(self):
        views = [
            ViewBox((20, 30, 50, 60), (100, 100), "x", "y"),
            ViewBox((40, 10, 70, 40), (100, 100), "y", "z"),
            ViewBox((15, 20, 35, 50), (100, 100), "z", "x"),
        ]
        result = intersect_views(views, (10, 0, -1), (20, 2, 1))
        for actual, expected in zip(result["bbox_min"], (12, 0.8, -0.7)):
            self.assertAlmostEqual(actual, expected)
        for actual, expected in zip(result["bbox_max"], (15, 1.2, -0.3)):
            self.assertAlmostEqual(actual, expected)

    def test_flip_maps_pixels_to_reversed_axis(self):
        views = [ViewBox((20, 0, 50, 100), (100, 100), "x", "y", flip_horizontal=True),
                 ViewBox((0, 0, 100, 100), (100, 100), "y", "z")]
        result = intersect_views(views, (0, 0, 0), (1, 1, 1))
        self.assertAlmostEqual(result["bbox_min"][0], 0.5)
        self.assertAlmostEqual(result["bbox_max"][0], 0.8)

    def test_disjoint_views_have_no_intersection(self):
        views = [ViewBox((0, 0, 20, 100), (100, 100), "x", "y"),
                 ViewBox((80, 0, 100, 100), (100, 100), "x", "z")]
        self.assertIsNone(intersect_views(views, (0, 0, 0), (1, 1, 1)))

    def test_rejects_unconstrained_axes_and_invalid_boxes(self):
        with self.assertRaises(ValueError):
            intersect_views([ViewBox((0, 0, 10, 10), (100, 100), "x", "y")], (0, 0, 0), (1, 1, 1))
        with self.assertRaises(ValueError):
            intersect_views([ViewBox((10, 0, 5, 10), (100, 100), "x", "y")], (0, 0, 0), (1, 1, 1))


if __name__ == "__main__":
    unittest.main()

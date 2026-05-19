from src.hierarchy.label_generator import generate_cluster_labels
from src.hierarchy.tree_builder import build_customer_hierarchy
from src.recommendation.recommender import generate_customer_recommendation


def test_generate_cluster_labels_contains_known_segments():
    """Validate cluster labels used for prediction output."""
    labels = generate_cluster_labels()

    assert labels[0] == "Regular Customers"
    assert labels[1] == "Premium Loyal Customers"
    assert labels[3] == "Dormant Customers"


def test_build_customer_hierarchy_groups_segments_by_activity():
    """Validate the high-level customer hierarchy shape."""
    hierarchy = build_customer_hierarchy()

    assert "Customers" in hierarchy
    assert "Premium Loyal Customers" in hierarchy["Customers"]["Active"]
    assert "Dormant Customers" in hierarchy["Customers"]["Inactive"]


def test_generate_customer_recommendation_returns_fallback_for_unknown_segment():
    """Validate recommendations for unknown segment labels."""
    recommendation = generate_customer_recommendation("New Segment")

    assert recommendation["summary"] == "Unknown customer type."
    assert recommendation["business_action"] == "Further analysis required."

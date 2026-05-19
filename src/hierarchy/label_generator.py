def generate_cluster_labels():
    """Return human-readable names for modeled customer clusters.

    Args:
        None.

    Returns:
        Dictionary mapping numeric cluster IDs to segment labels.
    """
    cluster_labels = {
        0: "Regular Customers",
        1: "Premium Loyal Customers",
        3: "Dormant Customers"
    }

    return cluster_labels

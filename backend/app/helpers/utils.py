def transform_to_range(valence, arousal):
    """
    Transform values from the range [0, 1] to [-1, 1]
    """
    transformed_valence = (valence - 0.5) * 2
    transformed_arousal = (arousal - 0.5) * 2
    return transformed_valence, transformed_arousal
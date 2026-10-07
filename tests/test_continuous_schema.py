from heptokens.data.continuous_schema import build_continuous_schema


def test_schema_uses_feature_names_not_input_order() -> None:
    schema = build_continuous_schema(
        collections={
            "tracks": {
                "inputs": [
                    "common/tracks/pt",
                    "common/tracks/eta",
                    "common/tracks/phi",
                    "common/tracks/d0",
                    "common/tracks/z0",
                    "atlas/tracks/qOverP",
                    "atlas/tracks/chiSquared",
                    "atlas/tracks/nDoF",
                ]
            }
        },
        object_order=["tracks"],
        event_token_specs=[
            {"input": "common/event/mu", "range": [0.0, 100.0]}
        ],
        type_ids={"tracks": 9},
        include_decoded_q8=True,
    )
    tracks = schema["objects"]["tracks"]
    assert tracks["groups"]["kinematics"] == [0, 1, 2, 5]
    assert tracks["groups"]["impact"] == [3, 4]
    assert tracks["groups"]["quality"] == [6, 7]
    assert schema["max_feature_dim"] == 8
    assert schema["feature_columns"]["decoded_q8"] == "decoded_continuous_features"
    assert schema["decoded_q8"]["event_inputs"].startswith("identical")

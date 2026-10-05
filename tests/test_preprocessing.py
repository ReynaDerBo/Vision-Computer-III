from src.data.csv_features import CLASS_MAP

def test_class_map():
    assert CLASS_MAP["benign"] == 0
    assert CLASS_MAP["malignant"] == 1
    assert CLASS_MAP["normal"] == 2

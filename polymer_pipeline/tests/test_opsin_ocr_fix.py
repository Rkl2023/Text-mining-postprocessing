import polymer_pipeline.layers.layer1_parse_text as lpt


def test_ocr_fix_positive():
    s = "2,6-bis(thiophen-2-yl)-4,8-bis(5-dodecylthiophen-2-yl)benzo[1,2-b:4,5-b0]dithiophene"
    out = lpt._preprocess_for_opsin_backbone_gentle(s)
    assert "benzo[1,2-b:4,5-b']dithiophene" in out


def test_ocr_fix_negative_no_colon():
    s = "foo[123]bar0"
    out = lpt._preprocess_for_opsin_backbone_gentle(s)
    assert out == s.replace("′", "'").replace("’", "'").strip(" .")


def test_ocr_fix_negative_legit_zero():
    s = "benzo[1,2-b:4,5-b10]dithiophene"
    out = lpt._preprocess_for_opsin_backbone_gentle(s)
    assert out == s.replace("′", "'").replace("’", "'").strip(" .")


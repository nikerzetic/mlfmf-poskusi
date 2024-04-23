def test_generate_path_features_for_function():
    import entries_extractor as ee    
    path = "D:\\Nik\\Projects\\mlfmf-poskusi\\stdlib\\code2vec\\train\\Agda.Builtin.Char_0003.dag"      
    G, leaves, name = ee.extract_graph(path)                                                       
    features = ee.generate_path_features_for_function(G, leaves, 8, 2)
    print(features)


def test_format_as_label():
    import entries_extractor as ee
    s = [
        "Data.Nat.Solver.+-*-Solver.-H_ 66",
        "Data.Bool.Properties._.IsIdempotentCommutativeMonoid 162",
        "Data.Nat.Properties.*-+-semiring 6512",
        "Data.Nat.Properties._._.¬Pn<1+v 6410",
    ]
    for w in s:
        print(ee.format_as_label(w))


if __name__ == "__main__":
    # test_generate_path_features_for_function()
    test_format_as_label()

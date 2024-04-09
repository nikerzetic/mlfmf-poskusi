def test_generate_path_features_for_function():
    import entries_extractor as ee    
    path = "D:\\Nik\\Projects\\mlfmf-poskusi\\stdlib\\code2vec\\train\\Agda.Builtin.Char_0003.dag"      
    G, leaves, name = ee.extract_graph(path)                                                       
    features = ee.generate_path_features_for_function(G, leaves, 8, 2)
    print(features)

if __name__ == "__main__":
    test_generate_path_features_for_function()

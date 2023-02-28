class GMA_Parameters():
    def __init__(self):

        # GMA parameters
        self.model = "atdn_vslam/checkpoints/gma-kitti.pth"
        self.dataset = "kitti"
        self.iters = 12
        self.num_heads = 1
        self.position_only = False
        self.position_and_content = False
        self.mixed_precision = True
        self.replace = False
        self.no_alpha = False
        self.no_residuals = False
        self.model_name = self.model
        self.path = "imgs"       
        
        self.dictionary = {
                           "model" : self.model,
                         "dataset" : self.dataset, 
                           "iters" : self.iters, 
                       "num_heads" : self.num_heads, 
                   "position_only" : self.position_only, 
            "position_and_content" : self.position_and_content, 
                 "mixed_precision" : self.mixed_precision,
                         "replace" : self.replace,
                        "no_alpha" : self.no_alpha,
                    "no_residuals" : self.no_residuals,
                            "imgs" : self.path,
                      "model_name" : self.model
        }

    def __contains__(self, key):
        return key in self.dictionary

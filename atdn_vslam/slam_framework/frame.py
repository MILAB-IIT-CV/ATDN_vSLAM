class Frame():
    """
    Frame class to store important keyframe attributes

    :param rgb_file_name: Name of the rgb file for a keyframe stored in the SLAM output path
    :param pred_pose: Predicted pose of the keyframe
    :param code: Latent space vector mapped to the keyframe
    """
    def __init__(   self,
                    rgb_file_name, 
                    pred_pose,
                    code = None) -> None:
        
        self.rgb_file_name = rgb_file_name
        self.pose = pred_pose
        self.embedding = code

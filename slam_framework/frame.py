

class Frame():
    def __init__(   self,
                    rgb_file_name, 
                    pred_pose,
                    code = None) -> None:
        
        self.rgb_file_name = rgb_file_name
        self.pose = pred_pose
        self.embedding = code

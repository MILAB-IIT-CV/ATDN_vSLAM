import torch


def line2matrix(pose : torch.Tensor):
    """
    Convert KITTI line format pose to homogenous matrix representation. Batched conversion is supported

    :param pose: Tensor of shape (12) if unbatched or (N, 12) if batched
    :return: Tensor of shape (3, 4) for unbatched (N, 3, 4) for batched input
    """
    shape = pose.shape
    if len(shape) == 1:
        matrix = torch.cat([pose.view((3, 4)), torch.tensor([[0, 0, 0, 1]])], dim=0)
    elif len(shape) == 2:
        seq, _, __ = shape
        matrix = torch.cat([pose.view((seq, 3, 4)), torch.tensor([[0, 0, 0, 1]]).repeat((seq, 1, 1))], dim=0)
    else:
        raise Exception("Invalid shape for line to matrix transformation")

    return matrix


def matrix2euler(R : torch.Tensor, convention : str = "yxz", device : str = "cpu"):
    """
    Convert rotation matrix to euler angles vector based on a given conversion convention

    :param R: Rotation matrix tensor of shape (3, 3)
    :param convention: Convention to convert to euler angles with. Learn more on Wikipedia_.
    :param device: Device to map output euler vector to.
    :return: Euler vector tensor of shape (3)

    .. _Wikipedia: https://en.wikipedia.org/wiki/Euler_angles
    """
    alpha, beta, gamma = None, None, None

    if convention == "yxz":
        alpha = torch.atan2(R[0, 2], R[2, 2])
        beta = torch.atan2(-R[1, 2], torch.sqrt(1-R[1, 2]**2))
        gamma = torch.atan2(R[1, 0], R[1, 1])
    elif convention == "yxy":
        alpha = torch.atan2(R[0, 1], R[2, 1])
        beta = torch.atan2(torch.sqrt(1-R[1, 1]**2), R[1, 1])
        gamma = torch.atan2(R[1, 0], -R[1, 2])

    euler = torch.tensor([alpha, beta, gamma], device=device)
    return euler


def euler2matrix(r : torch.Tensor, convention : str = "yxz", device = "cuda") -> torch.Tensor:
    """
    Convert euler angles to rotation matrix based on the conversion convention

    :param r: Euler angles tensor of shape (3)
    :param convention: Conversion convention for euler angles. Learn more on Wikipedia_.
    :param device: Device to map output matrix to.
    :return: Rotation matrix Tensor of shape (3, 3)

    .. _Wikipedia: https://en.wikipedia.org/wiki/Euler_angles
    """
    c1 = torch.cos(r[0])
    c2 = torch.cos(r[1])
    c3 = torch.cos(r[2])

    s1 = torch.sin(r[0])
    s2 = torch.sin(r[1])
    s3 = torch.sin(r[2])
    
    R = None

    if convention == "yxz":
        R = torch.tensor([  [c1*c3 + s1*s2*s3,  c3*s1*s2 - c1*s3,   c2*s1],
                            [c2*s3,             c2*c3,              -s2],
                            [c1*s2*s3 - c3*s1,  c1*c3*s2+s1*s3,     c1*c2]], device=device)
    elif convention == "xyx":
        R = torch.tensor([  [c2,                s2*s3,          c3*s2],
                            [s1*s2,             c1*c3-c2*s1*s3, -c3*s3-c2*c3*s1],
                            [-c1*s2,            c3*s1+c1*c2*s3, c1*c2*c3-s1*s3]], device=device)
    elif convention == "yxy":
        R = torch.tensor([  [c1*c3-c2*s1*s3,    s2*s1,          c2*s1*s1+c1*s3],
                            [s2*s3,             c2,             -s2*c3],
                            [-c3*s1-c2*c1*s3,   s2*c1,          c2*c1*c3-s1*s3]], device=device)

    if R is None:
        raise Exception("Convention " + str(convention) + " is not supported")

    return R


def transform(rot : torch.Tensor, tr : torch.Tensor, device="cpu"):
    """
    Transform Euler angles vector and translation vector to homogenous matrix

    :param rot: Euler vector tensor of shape (3).
    :param tr: Translation vector tensor of shape (3).
    :param device: Device to map output to.
    :return: Homogenous transformation matrix of shape (4, 4)
    """
    # Euler to matrix rotation representation
    rot = euler2matrix(rot, device=device)

    # Concatenating rotation matrix and translation vector
    mat = torch.cat([rot, tr.unsqueeze(1).to(device)], dim=1)

    # Adding the extra row for homogenous matrix
    mat = torch.cat([mat, torch.tensor([[0, 0, 0, 1]], device=device)], dim=0)

    return mat


# TODO Implement Homogenous Matrix to Euler+Translation conversion


def rel2abs(rotations : torch.Tensor, translations : torch.Tensor):
    """
    Convert relative rotation(s) and translation(s) vector representation to absolute scale pose(s)

    :param rotations: Euler angles rotation or rotations to convert
    :param translations: Translation vectors to convert
    :return: Absolute scale pose(s)
    """
    homogenous = []

    instance_num = len(rotations)
    for i in range(instance_num):
        homogenous.append(transform(rotations[i], translations[i]))

    global_scale = [torch.eye(4, dtype=homogenous[0].dtype)]
    for i in range(0, instance_num):
        global_scale.append(torch.matmul(global_scale[i], homogenous[i]))
        
    global_scale = torch.stack(global_scale, dim=0)
    #global_pos = global_scale[:, :3, -1]
    
    return global_scale


import torch


def line2matrix(pose):
    shape = pose.shape
    if len(shape) == 1:
        matrix = torch.cat([matrix.view((3, 4)), torch.tensor([[0, 0, 0, 1]])], dim=0)
    elif len(shape) == 2:
        seq, _, __ = shape
        matrix = torch.cat([matrix.view((seq, 3, 4)), torch.tensor([[0, 0, 0, 1]]).repeat((seq, 1, 1))], dim=0)
    else:
        raise Exception("Invalid shape for line to matrix transformation")

    return matrix


def euler2matrix(r : torch.Tensor, convention="yxz", device="cuda") -> torch.Tensor:
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


def matrix2euler(R, convention="yxz", device="cpu"):
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


def transform(rot : torch.Tensor, tr, device="cpu"):
    # Euler to matrix rotation representation
    rot = euler2matrix(rot, device=device)

    # Concatenating rotation matrix and translation vector
    mat = torch.cat([rot, tr.unsqueeze(1).to(device)], dim=1)

    # Adding the extra row for homogenous matrix
    mat = torch.cat([mat, torch.tensor([[0, 0, 0, 1]], device=device)], dim=0)

    return mat


# TODO Implement Homogenous Matrix to Euler+Translation conversion


def rel2abs(rotations, translations):
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


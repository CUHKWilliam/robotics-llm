import numpy as np
import mujoco
import mujoco_py

def check_joint_limits(q, model):
    """Check if the joints is under or above its limits"""
    for i in range(len(q)):
        q[i] = max(model.jnt_range[i][0], min(q[i], model.jnt_range[i][1]))

def circle(t: float, r: float, h: float, k: float, f: float) -> np.ndarray:
    """Return the (x, y) coordinates of a circle with radius r centered at (h, k)
    as a function of time t and frequency f."""
    x = r * np.cos(2 * np.pi * f * t) + h
    y = r * np.sin(2 * np.pi * f * t) + k
    z = 0.5
    return np.array([x, y, z])


initial_rot_vec = None
def calculate_ik(model, data, goal, env):
    # goal = np.array([-0.750,-0.025,1.8])
    goal = np.array([0, 0., 2.6])
    jac1 = np.zeros((6, model.nv)) #translation jacobian
    jac2 = np.zeros((6, model.nv)) #rotational jacobian
    step_size = 0.1
    tol = 0.02
    timeout = 100
    end_effector_id1 = model.site_name2id('end_effector')
    end_effector_id2 = model.site_name2id('end_effector')
    current_pose1 = data.site_xpos[end_effector_id1]
    current_pose2 = data.site_xpos[end_effector_id2]
    
    ee_quat = np.array([0, 0, 0, 0]).astype(np.float64)
    ee_ori = np.array([0, 0, 0]).astype(np.float64)

    mujoco_py.functions.mju_mat2Quat(ee_quat, data.site_xmat[end_effector_id1])
    mujoco.mju_quat2Vel(ee_ori, ee_quat, 1.0)
    tgt_ori = goal - (current_pose1 + current_pose2) / 2.
    ee_ori = ee_ori / np.linalg.norm(ee_ori)
    tgt_ori[-1] = 0.
    tgt_ori = tgt_ori / np.linalg.norm(tgt_ori)

    ## TODO
    tgt_ori = np.array([0., 0., 0.])
    error_ori = tgt_ori - ee_ori
    error_ori = error_ori / np.linalg.norm(error_ori)

    error = np.subtract(goal, (current_pose1 + current_pose2) / 2.) 
    error = np.concatenate([error, error_ori * 0.3])
    cnt = 0
    qpos0 = data.qpos
    # if True:
    while (np.linalg.norm(error) >= tol) and cnt < timeout: 
        #calculate jacobian

        mujoco_py.functions.mj_jacSite(model, data, jac1[:3].reshape(-1), jac1[3:].reshape(-1), end_effector_id1)
        mujoco_py.functions.mj_jacSite(model, data, jac2[:3].reshape(-1), jac2[3:].reshape(-1), end_effector_id2)
        jac = (jac1 + jac2) / 2.

        damping = 0.3
        diag = damping * np.eye(6)
        dq = jac.T @ np.linalg.solve(jac @ jac.T + diag, error)
        q = data.qpos.copy()
        mujoco_py.functions.mj_integratePos(model, q, dq, 0.02)
        data.qpos[:] = q.copy()


        # jacp = jac# [:3]
        # error = error#[:3]

        ## Gradient Descent 
        # grad =  0.5 * jacp.T @ error
        # data.qpos[:] += step_size * grad

        ## Gaussian-Newton
        # product = jacp.T @ jacp
        # if np.isclose(np.linalg.det(product), 0):
        #     j_inv = np.linalg.pinv(product) @ jacp.T
        # else:
        #     j_inv = np.linalg.inv(product) @ jacp.T
        # delta_q = j_inv @ error
        # data.qpos[:] += step_size * delta_q

        ## 
        # n = jacp.shape[1]
        # I = np.identity(n)
        # damping = 0.01
        # product = jacp.T @ jacp + damping * I
        # if np.isclose(np.linalg.det(product), 0):
        #     j_inv = np.linalg.pinv(product) @ jacp.T
        # else:
        #     j_inv = np.linalg.inv(product) @ jacp.T
        # delta_q = j_inv @ error
        # data.qpos[:] += step_size * delta_q
        

        check_joint_limits(data.qpos, model)
        mujoco_py.functions.mj_forward(model, data) 
        #calculate new error
        pos1 = data.site_xpos[end_effector_id1]
        pos2 = data.site_xpos[end_effector_id2]
        error = np.subtract(goal, (pos1 + pos2) / 2.) 
        
        mujoco_py.functions.mju_mat2Quat(ee_quat, data.site_xmat[end_effector_id1])
        mujoco.mju_quat2Vel(ee_ori, ee_quat, 1.0)
        tgt_ori = goal - (pos1 + pos2) / 2.
        tgt_ori[-1] = 0
        tgt_ori = tgt_ori / np.linalg.norm(tgt_ori)

        ## TODO:
        tgt_ori = np.array([0., -1., 0.])
        ee_ori = ee_ori / np.linalg.norm(ee_ori)
        error_ori = tgt_ori - ee_ori
        error_ori = error_ori / np.linalg.norm(error_ori)

        error = np.concatenate([error, error_ori * 0.3]) 
        print("error:", error)
        cnt += 1
    if cnt >= timeout:
        success = False
    else:
        success = True
    return data.qpos.copy(), success
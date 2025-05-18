import math
import numpy as np

class Quaternion:
   def __init__(self, x=0, y=0, z=0, w=1):
      self.m_data = [x, y, z, w]

   def __getitem__(self, index):
        return self.m_data[index]
   
   def __truediv__(self, other):
      result = Quaternion(
         float(self.m_data[0]) / other,
         float(self.m_data[1]) / other,
         float(self.m_data[2]) / other,
         float(self.m_data[3]) / other
      )
      return result
   
   def l2_norm(self):
      """
      计算四元数的 L2 范数（模长）。

      返回:
         float: 四元数的 L2 范数。
      """
      square_norm = (
         self.m_data[0] * self.m_data[0] +
         self.m_data[1] * self.m_data[1] +
         self.m_data[2] * self.m_data[2] +
         self.m_data[3] * self.m_data[3]
      )
      return math.sqrt(square_norm)

def axis_angle_to_quaternion(in_axis, theta):
   # normalize rotation axis
   norm = np.linalg.norm(in_axis)
   if norm < 1e-3:
      print("Warning: axis is zero so return identity quaternion")
      return Quaternion()

   axis = in_axis / norm

   # 计算四元数
   cos_theta_2 = math.cos(theta / 2)
   sin_theta_2 = math.sin(theta / 2)
   w = cos_theta_2
   x = axis[0] * sin_theta_2
   y = axis[1] * sin_theta_2
   z = axis[2] * sin_theta_2

   return Quaternion(x, y, z, w)

def quaternion_to_rotation_matrix(quaternion):
   # normalize quaternion
   norm = quaternion.l2_norm()
   if norm < 1e-3:
      print("Warning: quaternion is zero so return identity rotation matrix")
      return np.eye(3)

   q = quaternion / norm
   x, y, z, w = q.m_data

   # 计算旋转矩阵
   rotation_matrix = np.array([
      [1 - 2 * (y**2 + z**2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
      [2 * (x * y + z * w), 1 - 2 * (x**2 + z**2), 2 * (y * z - x * w)],
      [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)]
   ])

   return rotation_matrix

def axis_angle_to_rotation_matrix(in_axis, theta):
   """
   Args:
      in_axis (np.ndarray): rotation aixs, shape (3,)
      theta (float): rotation angle

   Return:
      np.ndarray: 3x3 rotation matrix
   """
   quaternion = axis_angle_to_quaternion(in_axis, theta)
   return quaternion_to_rotation_matrix(quaternion)


if __name__ == "__main__":
   axis = np.array([1.0, 0.0, 0.0])
   theta = np.pi / 2
   
   rotation_matrix_direct = axis_angle_to_rotation_matrix(axis, theta)
   print("Rotation matrix (direct):")
   print(rotation_matrix_direct)
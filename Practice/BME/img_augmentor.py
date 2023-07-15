import Augmentor

p = Augmentor.Pipeline("F:\\eyes\\data_R_L\\adult\\R")
p.ground_truth('blur')
p.ground_truth('blur')
p.ground_truth('blur')
p.ground_truth('blur')

p.rotate(probability=0.3, max_left_rotation=16, max_right_rotation=16)  # 旋转
p.flip_left_right(probability=0.5)  # 左右翻转
p.flip_top_bottom(probability=0.5)  # 上下翻转
p.rotate180(probability=0.3)  # 翻转180
p.zoom(probability=0.7, min_factor=1.1, max_factor=1.6)  #
p.zoom_random(probability=0.7, percentage_area=0.88)
p.sample(1000)

import torch
import numpy as np
from im2scene.common import interpolate_sphere
from torchvision.utils import save_image, make_grid
import imageio
from math import sqrt
from os import makedirs
from os.path import join


class Renderer(object):
    '''  Render class for GIRAFFE.

    It provides functions to render the representation.

    Args:
        model (nn.Module): trained GIRAFFE model
        device (device): pytorch device
    '''

    def __init__(self, model, device=None):
        self.model = model.to(device)
        gen = self.model.generator_test
        if gen is None:
            gen = self.model.generator
        gen.eval()
        self.generator = gen

        # sample temperature; only used for visualiations
        self.sample_tmp = 0.65

    def set_random_seed(self):
        torch.manual_seed(0)
        np.random.seed(0)

    def render_full_visualization(self, img_out_path,
                                  render_program=['object_rotation']):
        for rp in render_program:
            if rp == 'object_rotation':
                self.set_random_seed()
                self.render_object_rotation(img_out_path)
            if rp == 'object_translation_horizontal':
                self.set_random_seed()
                self.render_object_translation_horizontal(img_out_path)
            if rp == 'object_translation_vertical':
                self.set_random_seed()
                self.render_object_translation_depth(img_out_path)
            if rp == 'interpolate_app':
                self.set_random_seed()
                self.render_interpolation(img_out_path)
            if rp == 'interpolate_app_bg':
                self.set_random_seed()
                self.render_interpolation_bg(img_out_path)
            if rp == 'interpolate_shape':
                self.set_random_seed()
                self.render_interpolation(img_out_path, mode='shape')
            if rp == 'object_translation_circle':
                self.set_random_seed()
                self.render_object_translation_circle(img_out_path)
            if rp == 'render_camera_elevation':
                self.set_random_seed()
                self.render_camera_elevation(img_out_path)
            if rp == 'render_add_cars':
                self.set_random_seed()
                self.render_add_objects_cars5(img_out_path)
            if rp == 'render_add_clevr10':
                self.set_random_seed()
                self.render_add_objects_clevr10(img_out_path)
            if rp == 'render_add_clevr6':
                self.set_random_seed()
                self.render_add_objects_clevr6(img_out_path)

    def render_object_rotation(self, img_out_path, batch_size=15, n_steps=32):
        gen = self.generator
        bbox_generator = gen.bounding_box_generator

        n_boxes = bbox_generator.n_boxes

        # Set rotation range
        is_full_rotation = (bbox_generator.rotation_range[0] == 0
                            and bbox_generator.rotation_range[1] == 1)
        n_steps = int(n_steps * 2) if is_full_rotation else n_steps
        r_scale = [0., 1.] if is_full_rotation else [0.1, 0.9]

        # Get Random codes and bg rotation
        latent_codes = gen.get_latent_codes(batch_size, tmp=self.sample_tmp)
        bg_rotation = gen.get_random_bg_rotation(batch_size)

        # Set Camera
        camera_matrices = gen.get_camera(batch_size=batch_size)
        s_val = [[0, 0, 0] for i in range(n_boxes)]
        t_val = [[0.5, 0.5, 0.5] for i in range(n_boxes)]
        r_val = [0. for i in range(n_boxes)]
        s, t, _ = gen.get_transformations(s_val, t_val, r_val, batch_size)

        out = []
        for step in range(n_steps):
            # Get rotation for this step
            r = [step * 1.0 / (n_steps - 1) for i in range(n_boxes)]
            r = [r_scale[0] + ri * (r_scale[1] - r_scale[0]) for ri in r]
            r = gen.get_rotation(r, batch_size)

            # define full transformation and evaluate model
            transformations = [s, t, r]
            with torch.no_grad():
                out_i = gen(batch_size, latent_codes, camera_matrices,
                            transformations, bg_rotation, mode='val')
            out.append(out_i.cpu())
        out = torch.stack(out)
        out_folder = join(img_out_path, 'rotation_object')
        makedirs(out_folder, exist_ok=True)
        self.save_video_and_images(
            out, out_folder, name='rotation_object',
            is_full_rotation=is_full_rotation,
            add_reverse=(not is_full_rotation))

    def render_object_translation_horizontal(self, img_out_path, batch_size=15,
                                             n_steps=32):
        gen = self.generator

        # Get values
        latent_codes = gen.get_latent_codes(batch_size, tmp=self.sample_tmp)
        bg_rotation = gen.get_random_bg_rotation(batch_size)
        camera_matrices = gen.get_camera(batch_size=batch_size)
        n_boxes = gen.bounding_box_generator.n_boxes
        s = [[0., 0., 0.]
             for i in range(n_boxes)]
        r = [0.5 for i in range(n_boxes)]

        if n_boxes == 1:
            t = []
            x_val = 0.5
        elif n_boxes == 2:
            t = [[0.5, 0.5, 0.]]
            x_val = 1.

        out = []
        for step in range(n_steps):
            i = step * 1.0 / (n_steps - 1)
            ti = t + [[x_val, i, 0.]]
            transformations = gen.get_transformations(s, ti, r, batch_size)
            with torch.no_grad():
                out_i = gen(batch_size, latent_codes, camera_matrices,
                            transformations, bg_rotation, mode='val')
            out.append(out_i.cpu())
        out = torch.stack(out)

        out_folder = join(img_out_path, 'translation_object_horizontal')
        makedirs(out_folder, exist_ok=True)
        self.save_video_and_images(
            out, out_folder, name='translation_horizontal',
            add_reverse=True)

    def render_object_translation_depth(self, img_out_path, batch_size=15,
                                        n_steps=32):
        gen = self.generator
        # Get values
        latent_codes = gen.get_latent_codes(batch_size, tmp=self.sample_tmp)
        bg_rotation = gen.get_random_bg_rotation(batch_size)
        camera_matrices = gen.get_camera(batch_size=batch_size)

        n_boxes = gen.bounding_box_generator.n_boxes
        s = [[0., 0., 0.]
             for i in range(n_boxes)]
        r = [0.5 for i in range(n_boxes)]

        if n_boxes == 1:
            t = []
            y_val = 0.5
        elif n_boxes == 2:
            t = [[0.4, 0.8, 0.]]
            y_val = 0.2

        out = []
        for step in range(n_steps):
            i = step * 1.0 / (n_steps - 1)
            ti = t + [[i, y_val, 0.]]
            transformations = gen.get_transformations(s, ti, r, batch_size)
            with torch.no_grad():
                out_i = gen(batch_size, latent_codes, camera_matrices,
                            transformations, bg_rotation, mode='val')
            out.append(out_i.cpu())
        out = torch.stack(out)
        out_folder = join(img_out_path, 'translation_object_depth')
        makedirs(out_folder, exist_ok=True)
        self.save_video_and_images(
            out, out_folder, name='translation_depth', add_reverse=True)

    def render_interpolation(self, img_out_path, batch_size=15, n_samples=6,
                             n_steps=32, mode='app'):
        gen = self.generator
        n_boxes = gen.bounding_box_generator.n_boxes

        # Get values
        z_shape_obj_1, z_app_obj_1, z_shape_bg_1, z_app_bg_1 = \
            gen.get_latent_codes(batch_size, tmp=self.sample_tmp)

        z_i = [
            gen.sample_z(
                z_app_obj_1.shape,
                tmp=self.sample_tmp) for j in range(n_samples)
        ]

        bg_rotation = gen.get_random_bg_rotation(batch_size)
        camera_matrices = gen.get_camera(batch_size=batch_size)

        if n_boxes == 1:
            t_val = [[0.5, 0.5, 0.5]]
        transformations = gen.get_transformations(
            [[0., 0., 0.] for i in range(n_boxes)],
            t_val,
            [0.5 for i in range(n_boxes)],
            batch_size
        )

        out = []
        for j in range(n_samples):
            z_i1 = z_i[j]
            z_i2 = z_i[(j+1) % (n_samples)]
            for step in range(n_steps):
                w = step * 1.0 / ((n_steps) - 1)
                z_ii = interpolate_sphere(z_i1, z_i2, w)
                if mode == 'app':
                    latent_codes = [z_shape_obj_1, z_ii, z_shape_bg_1,
                                    z_app_bg_1]
                else:
                    latent_codes = [z_ii, z_app_obj_1, z_shape_bg_1,
                                    z_app_bg_1]
                with torch.no_grad():
                    out_i = gen(batch_size, latent_codes, camera_matrices,
                                transformations, bg_rotation, mode='val')
                out.append(out_i.cpu())
        out = torch.stack(out)

        # Save Video
        out_folder = join(img_out_path, 'interpolate_%s' % mode)
        makedirs(out_folder, exist_ok=True)
        self.save_video_and_images(
            out, out_folder, name='interpolate_%s' % mode,
            is_full_rotation=True)

    def render_interpolation_bg(self, img_out_path, batch_size=15, n_samples=6,
                                n_steps=32, mode='app'):
        gen = self.generator
        n_boxes = gen.bounding_box_generator.n_boxes

        # Get values
        z_shape_obj_1, z_app_obj_1, z_shape_bg_1, z_app_bg_1 = \
            gen.get_latent_codes(batch_size, tmp=self.sample_tmp)

        z_i = [
            gen.sample_z(
                z_app_bg_1.shape,
                tmp=self.sample_tmp) for j in range(n_samples)
        ]

        bg_rotation = gen.get_random_bg_rotation(batch_size)
        camera_matrices = gen.get_camera(batch_size=batch_size)

        if n_boxes == 1:
            t_val = [[0.5, 0.5, 0.5]]
        transformations = gen.get_transformations(
            [[0., 0., 0.] for i in range(n_boxes)],
            t_val,
            [0.5 for i in range(n_boxes)],
            batch_size
        )

        out = []
        for j in range(n_samples):
            z_i1 = z_i[j]
            z_i2 = z_i[(j+1) % (n_samples)]
            for step in range(n_steps):
                w = step * 1.0 / ((n_steps) - 1)
                z_ii = interpolate_sphere(z_i1, z_i2, w)
                if mode == 'app':
                    latent_codes = [z_shape_obj_1, z_app_obj_1, z_shape_bg_1,
                                    z_ii]
                else:
                    latent_codes = [z_shape_obj_1, z_app_obj_1, z_ii,
                                    z_app_bg_1]
                with torch.no_grad():
                    out_i = gen(batch_size, latent_codes, camera_matrices,
                                transformations, bg_rotation, mode='val')
                out.append(out_i.cpu())
        out = torch.stack(out)

        # Save Video
        out_folder = join(img_out_path, 'interpolate_bg_%s' % mode)
        makedirs(out_folder, exist_ok=True)
        self.save_video_and_images(
            out, out_folder, name='interpolate_bg_%s' % mode,
            is_full_rotation=True)

    def render_object_translation_circle(self, img_out_path, batch_size=15,
                                         n_steps=32):
        gen = self.generator

        # Disable object sampling
        sample_object_existance = gen.sample_object_existance
        gen.sample_object_existance = False

        # Get values
        latent_codes = gen.get_latent_codes(batch_size, tmp=self.sample_tmp)
        bg_rotation = gen.get_random_bg_rotation(batch_size)
        camera_matrices = gen.get_camera(batch_size=batch_size)
        n_boxes = gen.bounding_box_generator.n_boxes

        s = [[0, 0, 0, ]
             for i in range(n_boxes)]
        r = [0 for i in range(n_boxes)]
        s10, t10, r10 = gen.get_random_transformations(batch_size)

        out = []
        for step in range(n_steps):
            i = step * 1.0 / (n_steps - 1)
            cos_i = (np.cos(2 * np.pi * i) * 0.5 + 0.5).astype(np.float32)
            sin_i = (np.sin(2 * np.pi * i) * 0.5 + 0.5).astype(np.float32)
            if n_boxes <= 2:
                t = [[0.5, 0.5, 0.] for i in range(n_boxes - 1)] + [
                    [cos_i, sin_i, 0]
                ]
                transformations = gen.get_transformations(s, t, r, batch_size)
            else:
                cos_i, sin_i = cos_i * 1.0 - 0.0, sin_i * 1. - 0.
                _, ti, _ = gen.get_transformations(
                    val_t=[[cos_i, sin_i, 0]], batch_size=batch_size)
                t10[:, -1:] = ti
                transformations = [s10, t10, r10]

            with torch.no_grad():
                out_i = gen(batch_size, latent_codes, camera_matrices,
                            transformations, bg_rotation, mode='val')
            out.append(out_i.cpu())
        out = torch.stack(out)

        gen.sample_object_existance = sample_object_existance

        # Save Video
        out_folder = join(img_out_path, 'translation_circle')
        makedirs(out_folder, exist_ok=True)
        self.save_video_and_images(out, out_folder, name='translation_circle',
                                   is_full_rotation=True)

    def render_camera_elevation(self, img_out_path, batch_size=15, n_steps=32):
        gen = self.generator
        n_boxes = gen.bounding_box_generator.n_boxes
        r_range = [0.1, 0.9]

        # Get values
        latent_codes = gen.get_latent_codes(batch_size, tmp=self.sample_tmp)
        bg_rotation = gen.get_random_bg_rotation(batch_size)
        transformations = gen.get_transformations(
            [[0., 0., 0.] for i in range(n_boxes)],
            [[0.5, 0.5, 0.5] for i in range(n_boxes)],
            [0.5 for i in range(n_boxes)],
            batch_size,
        )

        out = []
        for step in range(n_steps):
            v = step * 1.0 / (n_steps - 1)
            r = r_range[0] + v * (r_range[1] - r_range[0])
            camera_matrices = gen.get_camera(val_v=r, batch_size=batch_size)
            with torch.no_grad():
                out_i = gen(
                    batch_size, latent_codes, camera_matrices, transformations,
                    bg_rotation, mode='val')
            out.append(out_i.cpu())
        out = torch.stack(out)

        out_folder = join(img_out_path, 'camera_elevation')
        makedirs(out_folder, exist_ok=True)
        self.save_video_and_images(out, out_folder, name='elevation_camera',
                                   is_full_rotation=False)

    def render_add_objects_cars5(self, img_out_path, batch_size=15):

        gen = self.generator

        # Get values
        z_shape_obj, z_app_obj, z_shape_bg, z_app_bg = gen.get_latent_codes(
            batch_size, tmp=self.sample_tmp)
        z_shape_obj = gen.sample_z(
            z_shape_obj[:, :1].repeat(1, 6, 1).shape, tmp=self.sample_tmp)
        z_app_obj = gen.sample_z(
            z_app_obj[:, :1].repeat(1, 6, 1).shape, tmp=self.sample_tmp)
        bg_rotation = gen.get_random_bg_rotation(batch_size)
        camera_matrices = gen.get_camera(val_v=0., batch_size=batch_size)

        s = [
            [-1., -1., -1.],
            [-1., -1., -1.],
            [-1., -1., -1.],
            [-1., -1., -1.],
            [-1., -1., -1.],
            [-1., -1., -1.],
        ]

        t = [
            [-0.7, -.8, 0.],
            [-0.7, 0.5, 0.],
            [-0.7, 1.8, 0.],
            [1.5, -.8, 0.],
            [1.5, 0.5, 0.],
            [1.5, 1.8, 0.],
        ]
        r = [
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
        ]
        outs = []
        for i in range(1, 7):
            transformations = gen.get_transformations(
                s[:i], t[:i], r[:i], batch_size)
            latent_codes = [z_shape_obj[:, :i], z_app_obj[:, :i], z_shape_bg,
                            z_app_bg]
            with torch.no_grad():
                out = gen(
                    batch_size, latent_codes, camera_matrices, transformations,
                    bg_rotation, mode='val').cpu()
            outs.append(out)
        outs = torch.stack(outs)
        idx = torch.arange(6).reshape(-1, 1).repeat(1, (128 // 6)).reshape(-1)
        outs = outs[[idx]]

        # import pdb; pdb.set_trace()
        out_folder = join(img_out_path, 'add_cars')
        makedirs(out_folder, exist_ok=True)
        self.save_video_and_images(outs, out_folder, name='add_cars',
                                   is_full_rotation=False, add_reverse=True)

    def render_add_objects_clevr10(self, img_out_path, batch_size=15):
        gen = self.generator

        # Disable object sampling
        sample_object_existance = gen.sample_object_existance
        gen.sample_object_existance = False

        n_steps = 6
        n_objs = 12

        # Get values
        z_shape_obj, z_app_obj, z_shape_bg, z_app_bg = gen.get_latent_codes(
            batch_size, tmp=self.sample_tmp)
        z_shape_obj = gen.sample_z(
            z_shape_obj[:, :1].repeat(1, n_objs, 1).shape, tmp=self.sample_tmp)
        z_app_obj = gen.sample_z(
            z_app_obj[:, :1].repeat(1, n_objs, 1).shape, tmp=self.sample_tmp)
        bg_rotation = gen.get_random_bg_rotation(batch_size)
        camera_matrices = gen.get_camera(val_v=0., batch_size=batch_size)

        s = [
            [0, 0, 0] for i in range(n_objs)
        ]
        t = []
        for i in range(n_steps):
            if i % 3 == 0:
                x = 0.0
            elif i % 3 == 1:
                x = 0.5
            else:
                x = 1

            if i in [0, 1, 2]:
                y = 0.
            else:
                y = 0.8
            t = t + [[x, y, 0], [x, y + 0.4, 0]]
        r = [
            0 for i in range(n_objs)
        ]
        out_total = []
        for i in range(2, n_objs + 1, 2):
            transformations = gen.get_transformations(
                s[:i], t[:i], r[:i], batch_size)
            latent_codes = [z_shape_obj[:, :i], z_app_obj[:, :i], z_shape_bg,
                            z_app_bg]
            with torch.no_grad():
                out = gen(
                    batch_size, latent_codes, camera_matrices, transformations,
                    bg_rotation, mode='val').cpu()
            out_total.append(out)
        out_total = torch.stack(out_total)
        idx = torch.arange(6).reshape(-1, 1).repeat(1, (128 // 6)).reshape(-1)
        outs = out_total[[idx]]

        gen.sample_object_existance = sample_object_existance

        out_folder = join(img_out_path, 'add_clevr_objects10')
        makedirs(out_folder, exist_ok=True)
        self.save_video_and_images(outs, out_folder, name='add_clevr10',
                                   is_full_rotation=False, add_reverse=True)

    def render_add_objects_clevr6(self, img_out_path, batch_size=15):

        gen = self.generator

        # Disable object sampling
        sample_object_existance = gen.sample_object_existance
        gen.sample_object_existance = False

        n_objs = 6
        # Get values
        z_shape_obj, z_app_obj, z_shape_bg, z_app_bg = gen.get_latent_codes(
            batch_size, tmp=self.sample_tmp)
        z_shape_obj = gen.sample_z(
            z_shape_obj[:, :1].repeat(1, n_objs, 1).shape, tmp=self.sample_tmp)
        z_app_obj = gen.sample_z(
            z_app_obj[:, :1].repeat(1, n_objs, 1).shape, tmp=self.sample_tmp)
        bg_rotation = gen.get_random_bg_rotation(batch_size)
        camera_matrices = gen.get_camera(val_v=0., batch_size=batch_size)

        s = [
            [0, 0, 0] for i in range(n_objs)
        ]
        t = []
        for i in range(n_objs):
            if i % 2 == 0:
                x = 0.2
            else:
                x = 0.8

            if i in [0, 1]:
                y = 0.
            elif i in [2, 3]:
                y = 0.5
            else:
                y = 1.
            t = t + [[x, y, 0]]
        r = [
            0 for i in range(n_objs)
        ]
        out_total = []
        for i in range(1, n_objs + 1):
            transformations = gen.get_transformations(
                s[:i], t[:i], r[:i], batch_size)
            latent_codes = [z_shape_obj[:, :i], z_app_obj[:, :i], z_shape_bg,
                            z_app_bg]
            with torch.no_grad():
                out = gen(
                    batch_size, latent_codes, camera_matrices, transformations,
                    bg_rotation, mode='val').cpu()
                out_total.append(out)
        out_total = torch.stack(out_total)
        idx = torch.arange(6).reshape(-1, 1).repeat(1, (128 // 6)).reshape(-1)
        outs = out_total[[idx]]

        gen.sample_object_existance = sample_object_existance

        out_folder = join(img_out_path, 'add_clevr_objects6')
        makedirs(out_folder, exist_ok=True)
        self.save_video_and_images(outs, out_folder, name='add_clevr6',
                                   is_full_rotation=False, add_reverse=True)

    ##################
    # Helper functions
    def write_video(self, out_file, img_list, n_row=5, add_reverse=False,
                    write_small_vis=True):
        n_steps, batch_size = img_list.shape[:2]
        nrow = n_row if (n_row is not None) else int(sqrt(batch_size))
        img = [(255*make_grid(img, nrow=nrow, pad_value=1.).permute(
            1, 2, 0)).cpu().numpy().astype(np.uint8) for img in img_list]
        if add_reverse:
            img += list(reversed(img))
        imageio.mimwrite(out_file, img, fps=30, quality=8)
        if write_small_vis:
            img = [(255*make_grid(img, nrow=batch_size, pad_value=1.).permute(
                1, 2, 0)).cpu().numpy().astype(
                    np.uint8) for img in img_list[:, :9]]
            if add_reverse:
                img += list(reversed(img))
            imageio.mimwrite(
                (out_file[:-4] + '_sm.mp4'), img, fps=30, quality=4)

    def save_video_and_images(self, imgs, out_folder, name='rotation_object',
                              is_full_rotation=False, img_n_steps=6,
                              add_reverse=False):

        # Save video
        out_file_video = join(out_folder, '%s.mp4' % name)
        self.write_video(out_file_video, imgs, add_reverse=add_reverse)

        # Save images
        n_steps, batch_size = imgs.shape[:2]
        if is_full_rotation:
            idx_paper = np.linspace(
                0, n_steps - n_steps // img_n_steps, img_n_steps
            ).astype(np.int)
        else:
            idx_paper = np.linspace(0, n_steps - 1, img_n_steps).astype(np.int)
        for idx in range(batch_size):
            img_grid = imgs[idx_paper, idx]
            save_image(make_grid(
                img_grid, nrow=img_n_steps, pad_value=1.), join(
                    out_folder, '%04d_%s.jpg' % (idx, name)))

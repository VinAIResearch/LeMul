import math

import networks
import numpy as np
import torch
import torch.nn as nn
import torchvision
import utils
from renderer.renderer import Renderer


EPS = 1e-7


class LeMul:
    def __init__(self, cfgs):
        self.model_name = cfgs.get("model_name", self.__class__.__name__)
        self.device = cfgs.get("device", "cpu")
        self.image_size = cfgs.get("image_size", 64)
        self.min_depth = cfgs.get("min_depth", 0.9)
        self.max_depth = cfgs.get("max_depth", 1.1)
        self.border_depth = cfgs.get("border_depth", (0.7 * self.max_depth + 0.3 * self.min_depth))
        self.xyz_rotation_range = cfgs.get("xyz_rotation_range", 90)
        self.xy_translation_range = cfgs.get("xy_translation_range", 0.1)
        self.z_translation_range = cfgs.get("z_translation_range", 0.1)
        self.lam_perc = cfgs.get("lam_perc", 1)
        self.lr = cfgs.get("lr", 1e-4)
        self.load_gt_depth = cfgs.get("load_gt_depth", False)
        self.renderer = Renderer(cfgs)

        # networks and optimizers
        self.netD = networks.EDDeconv(cin=3, cout=1, nf=64, zdim=256, activation=None)
        self.netA = networks.EDDeconv(cin=3, cout=3, nf=64, zdim=256)
        self.netL = networks.Encoder(cin=3, cout=4, nf=32)
        self.netV = networks.Encoder(cin=3, cout=6, nf=32)
        self.netC = networks.ConfNet(cin=3, cout=1, nf=64, zdim=128)
        self.netC2 = networks.ConfNet(cin=6, cout=1, nf=64, zdim=128)
        self.network_names = [k for k in vars(self) if "net" in k]
        self.run_finetune = cfgs.get("run_finetune", False)
        if self.run_finetune:
            self.net_optimize_names = ["netA", "netC", "netL"]
        else:
            self.net_optimize_names = self.network_names

        """Albedo loss's kernels"""
        # ====================================================#
        neighbor_kernels_list = []
        self.k_height = 3
        self.k_width = 3
        kernel_template = np.zeros((self.k_height, self.k_width))
        kernel_template[self.k_height // 2, self.k_width // 2] = 1
        for i in range(self.k_height):
            for j in range(self.k_width):
                tmp = kernel_template.copy()
                tmp[i, j] = -1
                neighbor_kernels_list.append(tmp)
        neighbor_kernels_list = np.stack(neighbor_kernels_list, 0)
        self.neighbor_kernel = torch.tensor(neighbor_kernels_list, dtype=torch.float).unsqueeze(1)
        self.neighbor_kernel.requires_grad = False
        self.neighbor_kernel = self.neighbor_kernel.to(self.device)
        # ====================================================#

        self.make_optimizer = lambda model: torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=self.lr, betas=(0.9, 0.999), weight_decay=5e-4
        )

        # other parameters
        self.PerceptualLoss = networks.PerceptualLoss(requires_grad=False)
        self.other_param_names = ["PerceptualLoss"]

        # depth rescaler: -1~1 -> min_deph~max_deph
        self.depth_rescaler = lambda d: (1 + d) / 2 * self.max_depth + (1 - d) / 2 * self.min_depth

    def init_optimizers(self):
        self.optimizer_names = []
        for net_name in self.net_optimize_names:
            optimizer = self.make_optimizer(getattr(self, net_name))
            optim_name = net_name.replace("net", "optimizer")
            setattr(self, optim_name, optimizer)
            self.optimizer_names += [optim_name]

    def load_model_state(self, cp):
        for k in cp:
            if k and k in self.network_names:
                getattr(self, k).load_state_dict(cp[k])

    def load_optimizer_state(self, cp):
        for k in cp:
            if k and k in self.optimizer_names:
                getattr(self, k).load_state_dict(cp[k])

    def get_model_state(self):
        states = {}
        for net_name in self.network_names:
            states[net_name] = getattr(self, net_name).state_dict()
        return states

    def get_optimizer_state(self):
        states = {}
        for optim_name in self.optimizer_names:
            states[optim_name] = getattr(self, optim_name).state_dict()
        return states

    def to_device(self, device):
        self.device = device
        for net_name in self.network_names:
            setattr(self, net_name, getattr(self, net_name).to(device))
        if self.other_param_names:
            for param_name in self.other_param_names:
                setattr(self, param_name, getattr(self, param_name).to(device))

    def set_train(self):
        for net_name in self.net_optimize_names:
            getattr(self, net_name).train()

    def set_eval(self):
        for net_name in self.network_names:
            getattr(self, net_name).eval()

    def photometric_loss(self, im1, im2, mask=None, conf_sigma=None):
        loss = (im1 - im2).abs()
        if conf_sigma is not None:
            loss = loss * 2 ** 0.5 / (conf_sigma + EPS) + (conf_sigma + EPS).log()
        if mask is not None:
            mask = mask.expand_as(loss)
            loss = (loss * mask).sum() / mask.sum()
        else:
            loss = loss.mean()
        return loss

    def cal_albedo_loss(self, input_im, recon_depth, recon_albedo, mask=None):
        # ========================================= #
        threshold = 0.5
        variance_c_square = 0.05
        variance_d_square = 2
        # input_im has shape BxCxHxW,
        input_im_neighbor_diff = torch.nn.functional.conv2d(
            input_im.view(-1, 1, input_im.shape[2], input_im.shape[3]),
            self.neighbor_kernel,
            padding=(self.k_height // 2, self.k_width // 2),
        )
        # output has shape: (B*C)xKxHxW, with K = k_height * k_width
        input_im_neighbor_diff = input_im_neighbor_diff.view(
            input_im.shape[0], 3, self.k_height * self.k_width, input_im.shape[2], input_im.shape[3]
        )
        input_im_neighbor_diff = input_im_neighbor_diff ** 2
        input_im_neighbor_diff = input_im_neighbor_diff.sum(1)
        # Sum over channels: (delta_r^2 + delta_g^2 + delta_b^2), output shape: B*K*H*W

        threshold_mask = (input_im_neighbor_diff < threshold).float()
        input_im_neighbor_diff = -input_im_neighbor_diff / (2 * variance_c_square)
        w_c_k = threshold_mask * torch.exp(input_im_neighbor_diff)
        # w_c_k has shape: BxKxHxW

        # depth has shape BxHxW
        depth_neighbor_diff = torch.nn.functional.conv2d(
            recon_depth.view(-1, 1, recon_depth.shape[1], recon_depth.shape[2]),
            self.neighbor_kernel,
            padding=(self.k_height // 2, self.k_width // 2),
        )
        # output has shape: BxKxHxW, with K = k_height * k_width
        depth_neighbor_diff = depth_neighbor_diff ** 2
        depth_neighbor_diff = -depth_neighbor_diff / (2 * variance_d_square)
        w_d_k = torch.exp(depth_neighbor_diff)

        albedo_neighbor_diff = torch.nn.functional.conv2d(
            recon_albedo.view(-1, 1, recon_albedo.shape[2], recon_albedo.shape[3]),
            self.neighbor_kernel,
            padding=(self.k_height // 2, self.k_width // 2),
        )
        # output has shape: (B*C)xKxHxW, with K = k_height * k_width

        # albedo has shape Bx3xHxW
        albedo_neighbor_diff = albedo_neighbor_diff.view(
            recon_albedo.shape[0], 3, self.k_height * self.k_width, recon_albedo.shape[2], recon_albedo.shape[3]
        )

        w_c_k = w_c_k.unsqueeze(1).repeat(1, 3, 1, 1, 1)  # B*3*K*H*W
        w_d_k = w_d_k.unsqueeze(1).repeat(1, 3, 1, 1, 1)  # B*3*K*H*W
        neighbor_weighted_diff = w_c_k * w_d_k * albedo_neighbor_diff
        sum_neighbor_weighted_diff = neighbor_weighted_diff.sum(2)
        sum_neighbor_weighted_diff = sum_neighbor_weighted_diff ** 2

        sum_neighbor_weighted_diff = sum_neighbor_weighted_diff.sum(1)  # B*K*H*W

        if mask is not None:
            sum_neighbor_weighted_diff = sum_neighbor_weighted_diff * mask

        albedo_loss = sum_neighbor_weighted_diff.mean()

        return albedo_loss

    def backward(self):
        for optim_name in self.optimizer_names:
            getattr(self, optim_name).zero_grad()
        self.loss_total.backward()
        for optim_name in self.optimizer_names:
            getattr(self, optim_name).step()

    def render(self, canon_albedo, canon_depth, canon_light, view):
        b = canon_albedo.shape[0]

        canon_light_a = canon_light[:, :1] / 2 + 0.5  # ambience term
        canon_light_b = canon_light[:, 1:2] / 2 + 0.5  # diffuse term
        canon_light_dxy = canon_light[:, 2:]
        canon_light_d = torch.cat([canon_light_dxy, torch.ones(b, 1).to(self.device)], 1)
        canon_light_d = canon_light_d / ((canon_light_d ** 2).sum(1, keepdim=True)) ** 0.5  # diffuse light direction

        canon_normal = self.renderer.get_normal_from_depth(canon_depth)
        canon_diffuse_shading = (canon_normal * canon_light_d.view(-1, 1, 1, 3)).sum(3).clamp(min=0).unsqueeze(1)
        canon_shading = canon_light_a.view(-1, 1, 1, 1) + canon_light_b.view(-1, 1, 1, 1) * canon_diffuse_shading
        canon_im = (canon_albedo / 2 + 0.5) * canon_shading * 2 - 1

        self.renderer.set_transform_matrices(view)
        recon_depth = self.renderer.warp_canon_depth(canon_depth)
        recon_normal = self.renderer.get_normal_from_depth(recon_depth)
        grid_2d_from_canon = self.renderer.get_inv_warped_2d_grid(recon_depth)
        recon_im = nn.functional.grid_sample(canon_im, grid_2d_from_canon, mode="bilinear")

        recon_albedo = nn.functional.grid_sample(canon_albedo, grid_2d_from_canon, mode="bilinear")

        margin = (self.max_depth - self.min_depth) / 2
        recon_im_mask = (
            recon_depth < self.max_depth + margin
        ).float()  # invalid border pixels have been clamped at max_depth+margin
        recon_im_mask_both = recon_im_mask.unsqueeze(1).detach()
        recon_im = recon_im * recon_im_mask_both

        return (
            grid_2d_from_canon,
            canon_diffuse_shading,
            recon_im,
            recon_depth,
            recon_normal,
            canon_im,
            canon_normal,
            recon_im_mask,
            recon_im_mask_both,
            recon_albedo,
        )

    def process(self, input_im):
        b, c, h, w = input_im.shape

        # predict canonical depth
        canon_depth_raw = self.netD(input_im).squeeze(1)  # BxHxW
        canon_depth = canon_depth_raw - canon_depth_raw.view(b, -1).mean(1).view(b, 1, 1)
        canon_depth = canon_depth.tanh()
        canon_depth = self.depth_rescaler(canon_depth)

        # clamp border depth
        depth_border = torch.zeros(1, h, w - 4).to(input_im.device)
        depth_border = nn.functional.pad(depth_border, (2, 2), mode="constant", value=1)
        canon_depth = canon_depth * (1 - depth_border) + depth_border * self.border_depth

        # predict canonical albedo
        canon_albedo = self.netA(input_im)  # Bx3xHxW

        # predict confidence map
        conf_sigma_l1, conf_sigma_percl = self.netC(input_im)  # Bx1xHxW

        # predict lighting
        canon_light = self.netL(input_im)  # Bx4

        # predict viewpoint transformation
        view = self.netV(input_im)
        view = torch.cat(
            [
                view[:, :3] * math.pi / 180 * self.xyz_rotation_range,
                view[:, 3:5] * self.xy_translation_range,
                view[:, 5:] * self.z_translation_range,
            ],
            1,
        )

        (
            grid_2d_from_canon,
            canon_diffuse_shading,
            recon_im,
            recon_depth,
            recon_normal,
            canon_im,
            canon_normal,
            recon_im_mask,
            recon_im_mask_both,
            recon_albedo,
        ) = self.render(canon_albedo, canon_depth, canon_light, view)

        loss_l1_im, loss_perc_im, albedo_loss, loss = self.loss(
            recon_im,
            input_im,
            recon_albedo=recon_albedo,
            recon_depth=recon_depth,
            recon_im_mask_both=recon_im_mask_both,
            conf_sigma_l1=conf_sigma_l1,
            conf_sigma_percl=conf_sigma_percl,
        )

        return (
            view,
            canon_depth_raw,
            canon_diffuse_shading,
            canon_albedo,
            recon_im,
            recon_depth,
            recon_normal,
            canon_im,
            canon_depth,
            canon_normal,
            conf_sigma_l1,
            conf_sigma_percl,
            canon_light,
            recon_im_mask,
            recon_im_mask_both,
            loss_l1_im,
            loss_perc_im,
            albedo_loss,
            loss,
            recon_albedo,
        )

    def loss(
        self,
        recon_im,
        input_im,
        recon_albedo=None,
        recon_depth=None,
        recon_im_mask_both=None,
        conf_sigma_l1=None,
        conf_sigma_percl=None,
    ):
        albedo_loss = self.cal_albedo_loss(input_im, recon_depth, recon_albedo, mask=recon_im_mask_both)
        loss_l1_im = self.photometric_loss(
            recon_im, input_im, mask=recon_im_mask_both, conf_sigma=conf_sigma_l1[:, :1]
        )
        loss_perc_im = self.PerceptualLoss(
            recon_im, input_im, mask=recon_im_mask_both, conf_sigma=conf_sigma_percl[:, :1]
        )
        loss = loss_l1_im + self.lam_perc * loss_perc_im + 0.5 * albedo_loss

        return loss_l1_im, loss_perc_im, albedo_loss, loss

    def forward(self, input):
        if self.load_gt_depth:
            input, input_support, depth_gt = input
        elif not self.run_finetune:
            input, input_support = input

        self.input_im = input.to(self.device) * 2.0 - 1.0
        b, c, h, w = self.input_im.shape

        (
            self.view,
            self.canon_depth_raw,
            self.canon_diffuse_shading,
            self.canon_albedo,
            self.recon_im,
            self.recon_depth,
            self.recon_normal,
            self.canon_im,
            self.canon_depth,
            self.canon_normal,
            self.conf_sigma_l1,
            self.conf_sigma_percl,
            self.canon_light,
            recon_im_mask,
            recon_im_mask_both,
            loss_l1_im,
            loss_perc_im,
            albedo_loss,
            self.loss_im,
            self.recon_albedo,
        ) = self.process(self.input_im)

        if not self.run_finetune:
            self.input_im_support = input_support.to(self.device) * 2.0 - 1.0
            (
                self.view_support,
                self.canon_depth_raw_support,
                self.canon_diffuse_shading_support,
                self.canon_albedo_support,
                self.recon_im_support,
                self.recon_depth_support,
                self.recon_normal_support,
                self.canon_im_support,
                self.canon_depth_support,
                self.canon_normal_support,
                self.conf_sigma_l1_support,
                self.conf_sigma_percl_support,
                self.canon_light_support,
                recon_im_mask_support,
                recon_im_mask_both_support,
                loss_l1_im_support,
                loss_perc_im_support,
                albedo_loss_support,
                self.loss_im_support,
                self.recon_albedo_support,
            ) = self.process(self.input_im_support)

            (
                _,
                _,
                self.recon_support_from_im,
                recon_depth_support_from_im,
                _,
                canon_im_support_from_im,
                _,
                _,
                recon_im_mask_both_support_from_im,
                recon_albedo_support_from_im,
            ) = self.render(self.canon_albedo, self.canon_depth, self.canon_light_support, self.view_support)
            conf_sigma_l1_norm_support, conf_sigma_percl_norm_support = self.netC2(
                torch.cat((self.input_im, self.input_im_support), 1)
            )  # Bx1xHxW
            _, _, _, self.loss_support_from_im = self.loss(
                self.recon_support_from_im,
                self.input_im_support,
                recon_albedo=recon_albedo_support_from_im,
                recon_depth=recon_depth_support_from_im,
                recon_im_mask_both=recon_im_mask_both_support_from_im,
                conf_sigma_l1=conf_sigma_l1_norm_support,
                conf_sigma_percl=conf_sigma_percl_norm_support,
            )

            (
                _,
                _,
                self.recon_im_from_support,
                recon_depth_im_from_support,
                _,
                canon_im_from_support,
                _,
                _,
                recon_im_mask_both_from_support,
                recon_albedo_im_from_support,
            ) = self.render(self.canon_albedo_support, self.canon_depth_support, self.canon_light, self.view)
            conf_sigma_l1_norm_im, conf_sigma_percl_norm_im = self.netC2(
                torch.cat((self.input_im_support, self.input_im), 1)
            )  # Bx1xHxW
            _, _, _, self.loss_im_from_support = self.loss(
                self.recon_im_from_support,
                self.input_im,
                recon_albedo=recon_albedo_im_from_support,
                recon_depth=recon_depth_im_from_support,
                recon_im_mask_both=recon_im_mask_both_from_support,
                conf_sigma_l1=conf_sigma_l1_norm_im,
                conf_sigma_percl=conf_sigma_percl_norm_im,
            )

            self.loss_total = (
                self.loss_im + self.loss_im_support + self.loss_support_from_im + self.loss_im_from_support
            )
        else:
            self.loss_total = self.loss_im

        metrics = {"loss": self.loss_total}

        # compute accuracy if gt depth is available
        if self.load_gt_depth:
            self.depth_gt = depth_gt[:, 0, :, :].to(self.input_im.device)
            self.depth_gt = (1 - self.depth_gt) * 2 - 1
            self.depth_gt = self.depth_rescaler(self.depth_gt)
            self.normal_gt = self.renderer.get_normal_from_depth(self.depth_gt)

            # mask out background
            mask_gt = (self.depth_gt < self.depth_gt.max()).float()
            mask_gt = (
                nn.functional.avg_pool2d(mask_gt.unsqueeze(1), 3, stride=1, padding=1).squeeze(1) > 0.99
            ).float()  # erode by 1 pixel
            mask_pred = (
                nn.functional.avg_pool2d(recon_im_mask.unsqueeze(1), 3, stride=1, padding=1).squeeze(1) > 0.99
            ).float()  # erode by 1 pixel
            mask = mask_gt * mask_pred
            self.acc_mae_masked = ((self.recon_depth - self.depth_gt).abs() * mask).view(b, -1).sum(1) / mask.view(
                b, -1
            ).sum(1)
            self.acc_mse_masked = (((self.recon_depth - self.depth_gt) ** 2) * mask).view(b, -1).sum(1) / mask.view(
                b, -1
            ).sum(1)
            self.sie_map_masked = utils.compute_sc_inv_err(self.recon_depth.log(), self.depth_gt.log(), mask=mask)
            self.acc_sie_masked = (self.sie_map_masked.view(b, -1).sum(1) / mask.view(b, -1).sum(1)) ** 0.5
            self.norm_err_map_masked = utils.compute_angular_distance(
                self.recon_normal[:b], self.normal_gt[:b], mask=mask
            )
            self.acc_normal_masked = self.norm_err_map_masked.view(b, -1).sum(1) / mask.view(b, -1).sum(1)

            metrics["SIE_masked"] = self.acc_sie_masked.mean()
            metrics["NorErr_masked"] = self.acc_normal_masked.mean()

        return metrics

    def visualize(self, logger, total_iter, max_bs=25):
        b, c, h, w = self.input_im.shape
        b0 = min(max_bs, b)

        # render rotations
        with torch.no_grad():
            v0 = torch.FloatTensor([-0.1 * math.pi / 180 * 60, 0, 0, 0, 0, 0]).to(self.input_im.device).repeat(b0, 1)
            canon_im_rotate = (
                self.renderer.render_yaw(self.canon_im[:b0], self.canon_depth[:b0], v_before=v0, maxr=90)
                .detach()
                .cpu()
                / 2.0
                + 0.5
            )  # (B,T,C,H,W)
            canon_normal_rotate = (
                self.renderer.render_yaw(
                    self.canon_normal[:b0].permute(0, 3, 1, 2), self.canon_depth[:b0], v_before=v0, maxr=90
                )
                .detach()
                .cpu()
                / 2.0
                + 0.5
            )  # (B,T,C,H,W)

        input_im = self.input_im[:b0].detach().cpu() / 2 + 0.5
        if not self.run_finetune:
            input_im_support = self.input_im_support[:b0].detach().cpu() / 2.0 + 0.5
        canon_albedo = self.canon_albedo[:b0].detach().cpu() / 2.0 + 0.5
        recon_albedo = self.recon_albedo[:b0].detach().cpu() / 2.0 + 0.5
        canon_im = self.canon_im[:b0].detach().cpu() / 2.0 + 0.5
        recon_im = self.recon_im[:b0].detach().cpu() / 2.0 + 0.5
        if not self.run_finetune:
            recon_im_support = self.recon_im_support[:b0].detach().cpu() / 2.0 + 0.5
        canon_depth_raw_hist = self.canon_depth_raw.detach().unsqueeze(1).cpu()
        canon_depth_raw = self.canon_depth_raw[:b0].detach().unsqueeze(1).cpu() / 2.0 + 0.5
        canon_depth = (
            ((self.canon_depth[:b0] - self.min_depth) / (self.max_depth - self.min_depth)).detach().cpu().unsqueeze(1)
        )
        recon_depth = (
            ((self.recon_depth[:b0] - self.min_depth) / (self.max_depth - self.min_depth)).detach().cpu().unsqueeze(1)
        )
        canon_diffuse_shading = self.canon_diffuse_shading[:b0].detach().cpu()
        canon_normal = self.canon_normal.permute(0, 3, 1, 2)[:b0].detach().cpu() / 2 + 0.5
        recon_normal = self.recon_normal.permute(0, 3, 1, 2)[:b0].detach().cpu() / 2 + 0.5
        conf_map_l1 = 1 / (1 + self.conf_sigma_l1[:b0, :1].detach().cpu() + EPS)
        conf_map_percl = 1 / (1 + self.conf_sigma_percl[:b0, :1].detach().cpu() + EPS)

        canon_im_rotate_grid = [
            torchvision.utils.make_grid(img, nrow=int(math.ceil(b0 ** 0.5)))
            for img in torch.unbind(canon_im_rotate, 1)
        ]  # [(C,H,W)]*T
        canon_im_rotate_grid = torch.stack(canon_im_rotate_grid, 0).unsqueeze(0)  # (1,T,C,H,W)
        canon_normal_rotate_grid = [
            torchvision.utils.make_grid(img, nrow=int(math.ceil(b0 ** 0.5)))
            for img in torch.unbind(canon_normal_rotate, 1)
        ]  # [(C,H,W)]*T
        canon_normal_rotate_grid = torch.stack(canon_normal_rotate_grid, 0).unsqueeze(0)  # (1,T,C,H,W)

        # write summary
        logger.add_scalar("Loss/loss_total", self.loss_total, total_iter)

        logger.add_histogram("Depth/canon_depth_raw_hist", canon_depth_raw_hist, total_iter)
        vlist = ["view_rx", "view_ry", "view_rz", "view_tx", "view_ty", "view_tz"]
        for i in range(self.view.shape[1]):
            logger.add_histogram("View/" + vlist[i], self.view[:, i], total_iter)

        def log_grid_image(label, im, nrow=int(math.ceil(b0 ** 0.5)), iter=total_iter):
            im_grid = torchvision.utils.make_grid(im, nrow=nrow)
            logger.add_image(label, im_grid, iter)

        log_grid_image("Image/input_image", input_im)
        if not self.run_finetune:
            log_grid_image("Image/input_image_support", input_im_support)
        log_grid_image("Image/canonical_albedo", canon_albedo)
        log_grid_image("Image/recon_albedo", recon_albedo)
        log_grid_image("Image/canonical_image", canon_im)
        log_grid_image("Image/recon_image", recon_im)
        if not self.run_finetune:
            log_grid_image("Image/recon_image_support", recon_im_support)
        log_grid_image("Image/recon_side", canon_im_rotate[:, 0, :, :, :])
        log_grid_image("Image/recon_side_2", canon_im_rotate[:, 1, :, :, :])

        log_grid_image("Depth/canonical_depth_raw", canon_depth_raw)
        log_grid_image("Depth/canonical_depth", canon_depth)
        log_grid_image("Depth/recon_depth", recon_depth)
        log_grid_image("Depth/canonical_diffuse_shading", canon_diffuse_shading)
        log_grid_image("Depth/canonical_normal", canon_normal)
        log_grid_image("Depth/recon_normal", recon_normal)

        logger.add_histogram("Image/canonical_albedo_hist", canon_albedo, total_iter)
        logger.add_histogram("Image/canonical_diffuse_shading_hist", canon_diffuse_shading, total_iter)

        log_grid_image("Conf/conf_map_l1", conf_map_l1)
        logger.add_histogram("Conf/conf_sigma_l1_hist", self.conf_sigma_l1[:, :1], total_iter)
        log_grid_image("Conf/conf_map_percl", conf_map_percl)
        logger.add_histogram("Conf/conf_sigma_percl_hist", self.conf_sigma_percl[:, :1], total_iter)

        logger.add_video("Image_rotate/recon_rotate", canon_im_rotate_grid, total_iter, fps=4)
        logger.add_video("Image_rotate/canon_normal_rotate", canon_normal_rotate_grid, total_iter, fps=4)

        # visualize images and accuracy if gt is loaded
        if self.load_gt_depth:
            depth_gt = (
                ((self.depth_gt[:b0] - self.min_depth) / (self.max_depth - self.min_depth)).detach().cpu().unsqueeze(1)
            )
            normal_gt = self.normal_gt.permute(0, 3, 1, 2)[:b0].detach().cpu() / 2 + 0.5
            sie_map_masked = self.sie_map_masked[:b0].detach().unsqueeze(1).cpu() * 1000
            norm_err_map_masked = self.norm_err_map_masked[:b0].detach().unsqueeze(1).cpu() / 100

            logger.add_scalar("Acc_masked/MAE_masked", self.acc_mae_masked.mean(), total_iter)
            logger.add_scalar("Acc_masked/MSE_masked", self.acc_mse_masked.mean(), total_iter)
            logger.add_scalar("Acc_masked/SIE_masked", self.acc_sie_masked.mean(), total_iter)
            logger.add_scalar("Acc_masked/NorErr_masked", self.acc_normal_masked.mean(), total_iter)

            log_grid_image("Depth_gt/depth_gt", depth_gt)
            log_grid_image("Depth_gt/normal_gt", normal_gt)
            log_grid_image("Depth_gt/sie_map_masked", sie_map_masked)
            log_grid_image("Depth_gt/norm_err_map_masked", norm_err_map_masked)

    def save_results(self, save_dir):
        b, c, h, w = self.input_im.shape

        with torch.no_grad():
            v0 = torch.FloatTensor([-0.1 * math.pi / 180 * 60, 0, 0, 0, 0, 0]).to(self.input_im.device).repeat(b, 1)
            canon_im_rotate = self.renderer.render_yaw(
                self.canon_im[:b], self.canon_depth[:b], v_before=v0, maxr=90, nsample=15
            )  # (B,T,C,H,W)
            canon_im_rotate = canon_im_rotate.clamp(-1, 1).detach().cpu() / 2 + 0.5
            canon_normal_rotate = self.renderer.render_yaw(
                self.canon_normal[:b].permute(0, 3, 1, 2), self.canon_depth[:b], v_before=v0, maxr=90, nsample=15
            )  # (B,T,C,H,W)
            canon_normal_rotate = canon_normal_rotate.clamp(-1, 1).detach().cpu() / 2 + 0.5

        input_im = self.input_im[:b].detach().cpu().numpy() / 2 + 0.5
        canon_albedo = self.canon_albedo[:b].detach().cpu().numpy() / 2 + 0.5
        canon_im = self.canon_im[:b].clamp(-1, 1).detach().cpu().numpy() / 2 + 0.5
        recon_im = self.recon_im[:b].clamp(-1, 1).detach().cpu().numpy() / 2 + 0.5
        if not self.run_finetune:
            recon_im_support = self.recon_im[b:].clamp(-1, 1).detach().cpu().numpy() / 2 + 0.5
        canon_depth = (
            ((self.canon_depth[:b] - self.min_depth) / (self.max_depth - self.min_depth))
            .clamp(0, 1)
            .detach()
            .cpu()
            .unsqueeze(1)
            .numpy()
        )
        recon_depth = (
            ((self.recon_depth[:b] - self.min_depth) / (self.max_depth - self.min_depth))
            .clamp(0, 1)
            .detach()
            .cpu()
            .unsqueeze(1)
            .numpy()
        )
        canon_diffuse_shading = self.canon_diffuse_shading[:b].detach().cpu().numpy()
        canon_normal = self.canon_normal[:b].permute(0, 3, 1, 2).detach().cpu().numpy() / 2 + 0.5
        recon_normal = self.recon_normal[:b].permute(0, 3, 1, 2).detach().cpu().numpy() / 2 + 0.5
        conf_map_l1 = 1 / (1 + self.conf_sigma_l1[:b, :1].detach().cpu().numpy() + EPS)
        conf_map_percl = 1 / (1 + self.conf_sigma_percl[:b, :1].detach().cpu().numpy() + EPS)
        view = self.view[:b].detach().cpu().numpy()

        canon_im_rotate_grid = [
            torchvision.utils.make_grid(img, nrow=int(math.ceil(b ** 0.5))) for img in torch.unbind(canon_im_rotate, 1)
        ]  # [(C,H,W)]*T
        canon_im_rotate_grid = torch.stack(canon_im_rotate_grid, 0).unsqueeze(0).numpy()  # (1,T,C,H,W)
        canon_normal_rotate_grid = [
            torchvision.utils.make_grid(img, nrow=int(math.ceil(b ** 0.5)))
            for img in torch.unbind(canon_normal_rotate, 1)
        ]  # [(C,H,W)]*T
        canon_normal_rotate_grid = torch.stack(canon_normal_rotate_grid, 0).unsqueeze(0).numpy()  # (1,T,C,H,W)

        sep_folder = True
        utils.save_images(save_dir, input_im, suffix="input_image", sep_folder=sep_folder)
        utils.save_images(save_dir, canon_albedo, suffix="canonical_albedo", sep_folder=sep_folder)
        utils.save_images(save_dir, canon_im, suffix="canonical_image", sep_folder=sep_folder)
        utils.save_images(save_dir, recon_im, suffix="recon_image", sep_folder=sep_folder)
        if not self.run_finetune:
            utils.save_images(save_dir, recon_im_support, suffix="recon_image_support", sep_folder=sep_folder)
        utils.save_images(save_dir, canon_depth, suffix="canonical_depth", sep_folder=sep_folder)
        utils.save_images(save_dir, recon_depth, suffix="recon_depth", sep_folder=sep_folder)
        utils.save_images(save_dir, canon_diffuse_shading, suffix="canonical_diffuse_shading", sep_folder=sep_folder)
        utils.save_images(save_dir, canon_normal, suffix="canonical_normal", sep_folder=sep_folder)
        utils.save_images(save_dir, recon_normal, suffix="recon_normal", sep_folder=sep_folder)
        utils.save_images(save_dir, conf_map_l1, suffix="conf_map_l1", sep_folder=sep_folder)
        utils.save_images(save_dir, conf_map_percl, suffix="conf_map_percl", sep_folder=sep_folder)
        utils.save_txt(save_dir, view, suffix="viewpoint", sep_folder=sep_folder)

        utils.save_videos(save_dir, canon_im_rotate_grid, suffix="image_video", sep_folder=sep_folder, cycle=True)
        utils.save_videos(save_dir, canon_normal_rotate_grid, suffix="normal_video", sep_folder=sep_folder, cycle=True)

        # save scores if gt is loaded
        if self.load_gt_depth:
            depth_gt = (
                ((self.depth_gt[:b] - self.min_depth) / (self.max_depth - self.min_depth))
                .clamp(0, 1)
                .detach()
                .cpu()
                .unsqueeze(1)
                .numpy()
            )
            normal_gt = self.normal_gt[:b].permute(0, 3, 1, 2).detach().cpu().numpy() / 2 + 0.5
            utils.save_images(save_dir, depth_gt, suffix="depth_gt", sep_folder=sep_folder)
            utils.save_images(save_dir, normal_gt, suffix="normal_gt", sep_folder=sep_folder)

            all_scores = torch.stack(
                [
                    self.acc_mae_masked.detach().cpu(),
                    self.acc_mse_masked.detach().cpu(),
                    self.acc_sie_masked.detach().cpu(),
                    self.acc_normal_masked.detach().cpu(),
                ],
                1,
            )
            if not hasattr(self, "all_scores"):
                self.all_scores = torch.FloatTensor()
            self.all_scores = torch.cat([self.all_scores, all_scores], 0)

    def save_scores(self, path):
        # save scores if gt is loaded
        if self.load_gt_depth:
            header = "MAE_masked, \
                      MSE_masked, \
                      SIE_masked, \
                      NorErr_masked"
            mean = self.all_scores.mean(0)
            std = self.all_scores.std(0)
            header = header + "\nMean: " + ",\t".join(["%.8f" % x for x in mean])
            header = header + "\nStd: " + ",\t".join(["%.8f" % x for x in std])
            utils.save_scores(path, self.all_scores, header=header)

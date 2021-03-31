import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import time
import random
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import optim, nn
from torchvision import transforms
from torchvision.utils import make_grid

from conf import pix2pix_settings as Settings
from models.Generator import Generator
from models.Discriminator import Discriminator
from utils.DataLoaders import get_dataloader


class Pix2PixMain(object):

    def __init__(self):

        # -----------------------------------
        # global
        # -----------------------------------
        np.random.seed(Settings.SEED)
        torch.manual_seed(Settings.SEED)
        random.seed(Settings.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(Settings.SEED)
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # -----------------------------------
        # model
        # -----------------------------------
        self.generator = Generator(in_c=Settings.IN_CHANNEL, out_c=Settings.OUT_CHANNEL,
                                   ngf=Settings.NGF).to(self.device)
        self.generator.apply(self.generator.weights_init)
        self.discriminator = Discriminator(in_c=Settings.IN_CHANNEL, out_c=Settings.OUT_CHANNEL,
                                           ndf=Settings.NDF, n_layers=Settings.DISCRIMINATOR_LAYER).to(self.device)
        self.discriminator.apply(self.discriminator.weights_init)
        print("model init done")

        # -----------------------------------
        # data
        # -----------------------------------
        train_transforms = transforms.Compose([transforms.Resize((Settings.INPUT_SIZE, Settings.INPUT_SIZE)),
                                               transforms.ToTensor(),
                                               transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

        data_prepare = get_dataloader(dataset_name=Settings.DATASET,
                                      batch_size=Settings.BATCH_SIZE,
                                      data_root=Settings.DATASET_ROOT,
                                      train_num_workers=Settings.TRAIN_NUM_WORKERS,
                                      transforms=train_transforms,
                                      val_num_workers=Settings.TEST_NUM_WORKERS)
        self.train_dataloader = data_prepare.train_dataloader
        self.test_dataloader = data_prepare.test_dataloader
        print("data init done.....")

        # -----------------------------------
        # optimizer and criterion
        # -----------------------------------
        self.optimG = optim.Adam([{"params": self.generator.parameters()}],
                                 lr=Settings.G_LR, betas=Settings.G_BETAS)
        self.optimD = optim.Adam([{"params": self.discriminator.parameters()}],
                                 lr=Settings.D_LR, betas=Settings.D_BETAS)

        self.criterion_l1loss = nn.L1Loss()
        self.criterion_BCE = nn.BCELoss()
        print("optimizer and criterion init done.....")

        # -----------------------------------
        # recorder
        # -----------------------------------
        self.recorder = {"errD_fake": list(),
                         "errD_real": list(),
                         "errG_l1loss": list(),
                         "errG_bce": list(),
                         "errG": list(),
                         "accD": list()}

        output_file = time.strftime("{}_%Y_%m_%d_%H_%M_%S".format("pix2pix"), time.localtime())
        self.output_root = os.path.join(Settings.OUTPUT_ROOT, output_file)
        os.makedirs(os.path.join(self.output_root, Settings.OUTPUT_MODEL_KEY))
        os.makedirs(os.path.join(self.output_root, Settings.OUTPUT_LOG_KEY))
        os.makedirs(os.path.join(self.output_root, Settings.OUTPUT_IMAGE_KEY))
        print("recorder init done.....")

    def __call__(self):

        print_steps = int(len(self.train_dataloader) * Settings.PRINT_FREQUENT)
        eval_steps = int(len(self.train_dataloader) * Settings.EVAL_FREQUENT)

        print("begin train.....")
        for epoch in range(1, Settings.EPOCHS+1):
            for step, batch in enumerate(self.train_dataloader):

                # train
                self.train_module(batch)

                # print
                self.print_module(epoch, step, print_steps)

                # val
                self.val_module(epoch, step, eval_steps)

            # save log
            self.log_save_module()

    def train_module(self, batch):
        self.generator.train()
        self.discriminator.train()

        input_images = None
        target_images = None
        if Settings.DATASET == "edge2shoes":
            input_images = batch["edge_images"].to(self.device)
            target_images = batch["color_images"].to(self.device)
        else:
            KeyError("DataSet {} doesn't exit".format(Settings.DATASET))

        # 判别器迭代
        self.optimD.zero_grad()
        true_image_d_pred = self.discriminator(input_images, target_images)
        true_images_label = torch.full(true_image_d_pred.shape,
                                       Settings.REAL_LABEL, dtype=torch.float32, device=self.device)
        errD_real_bce = self.criterion_BCE(true_image_d_pred, true_images_label)
        errD_real_bce.backward()

        fake_images = self.generator(input_images)
        fake_images_d_pred = self.discriminator(input_images, fake_images.detach())
        fake_images_label = torch.full(fake_images_d_pred.shape,
                                       Settings.FAKE_LABEL, dtype=torch.float32, device=self.device)
        errD_fake_bce = self.criterion_BCE(fake_images_d_pred, fake_images_label)
        errD_fake_bce.backward()
        self.optimD.step()

        real_image_pred_true_num = ((true_image_d_pred > 0.5) == true_images_label).sum().float()
        fake_image_pred_true_num = ((fake_images_d_pred > 0.5) == fake_images_label).sum().float()

        accD = (real_image_pred_true_num + fake_image_pred_true_num) / \
               (true_images_label.numel() + fake_images_label.numel())

        # 生成器迭代
        self.optimG.zero_grad()
        fake_images_d_pred = self.discriminator(input_images, fake_images)
        true_images_label = torch.full(fake_images_d_pred.shape,
                                       Settings.REAL_LABEL, dtype=torch.float32, device=self.device)
        errG_bce = self.criterion_BCE(fake_images_d_pred, true_images_label)
        errG_l1loss = self.criterion_l1loss(fake_images, target_images)

        errG = errG_bce + errG_l1loss * Settings.L1_LOSS_LAMUDA
        errG.backward()
        self.optimG.step()

        # recorder
        self.recorder["errD_real"].append(errD_real_bce.item())
        self.recorder["errD_fake"].append(errD_fake_bce.item())
        self.recorder["errG_l1loss"].append(errG_l1loss.item())
        self.recorder["errG_bce"].append(errG_bce.item())
        self.recorder["errG"].append(errG.item())
        self.recorder["accD"].append(accD)

    def val_module(self, epoch, step, eval_steps):

        def apply_dropout(m):
            if type(m) == nn.Dropout:
                m.train()

        if (step+1) % eval_steps == 0:

            output_images = None
            output_count = 0

            self.generator.eval()
            self.discriminator.eval()

            # 启用dropout
            if Settings.USING_DROPOUT_DURING_EVAL:
                self.generator.apply(apply_dropout)
                self.discriminator.apply(apply_dropout)

            for eval_step, eval_batch in enumerate(self.test_dataloader):

                input_images = eval_batch["edge_images"].to(self.device)
                target_images = eval_batch["color_images"]

                pred_images = self.generator(input_images).detach().cpu()

                output_image = torch.cat([input_images.cpu(), target_images, pred_images], dim=3)

                if output_images is None:
                    output_images = output_image
                else:
                    output_images = torch.cat([output_images, output_image], dim=0)

                if output_images.shape[0] == 40:

                    output_images = make_grid(output_images,
                                              padding=2,
                                              normalize=True,
                                              nrow=Settings.CONSTANT_FEATURE_DIS_LEN).numpy()
                    output_images = np.array(np.transpose(output_images, (1, 2, 0))*255, dtype=np.uint8)
                    output_images = Image.fromarray(output_images)
                    output_images.save(os.path.join(self.output_root, Settings.OUTPUT_IMAGE_KEY,
                                                    "epoch_{}_step_{}_count_{}.jpg".format(epoch, step, output_count)))

                    output_count += 1
                    output_images = None

            self.model_save_module(epoch, step)
            self.log_save_module()

    def print_module(self, epoch, step, print_steps):
        if (step+1) % print_steps == 0:
            print("[{}/{}]\t [{}/{}]\t ".format(epoch, Settings.EPOCHS, step+1, len(self.train_dataloader)), end=" ")

            for key in self.recorder:
                print("[{}:{}]\t".format(key, self.recorder[key][-1]), end=" ")

            print(" ")

    def model_save_module(self, epoch, step):
        torch.save(self.generator.state_dict(),
                   os.path.join(self.output_root, Settings.OUTPUT_MODEL_KEY,
                                "pix2pix_generator_epoch_{}_step_{}.pth".format(epoch, step)))
        torch.save(self.discriminator.state_dict(),
                   os.path.join(self.output_root, Settings.OUTPUT_MODEL_KEY,
                                "pix2pix_discriminator_epoch_{}_step_{}.pth".format(epoch, step)))

    def log_save_module(self):
        # 保存记录
        with open(os.path.join(self.output_root, Settings.OUTPUT_LOG_KEY, "log.txt"), "w") as f:
            for item_ in range(len(self.recorder["accD"])):
                for key in self.recorder:
                    f.write("{}:{}\t".format(key, self.recorder[key][item_]))
                f.write("\n")

        # 保存图表
        for key in self.recorder:
            plt.figure(figsize=(10, 5))
            plt.title("{} During Training".format(key))
            plt.plot(self.recorder[key], label=key)
            plt.xlabel("iterations")
            plt.ylabel("value")
            plt.legend()
            if "acc" in key:
                plt.yticks(np.arange(0, 1, 0.5))
            plt.savefig(os.path.join(self.output_root, Settings.OUTPUT_LOG_KEY, "{}.jpg".format(key)))

        plt.close("all")

    def learning_rate_decay_module(self, epoch):
        if epoch % Settings.LR_DECAY_EPOCHS == 0:
            for param_group in self.optimD.param_groups:
                param_group["lr"] *= 0.2
            for param_group in self.optimG.param_groups:
                param_group["lr"] *= 0.2


if __name__ == "__main__":
    Pix2Pix = Pix2PixMain()
    Pix2Pix()

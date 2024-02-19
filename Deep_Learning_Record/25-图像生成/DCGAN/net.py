import torch
import torch.nn as nn
from torchvision import utils, datasets, transforms
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import os

import matplotlib as mpl
mpl.rcParams['animation.embed_limit'] = 40.0  # 单位为MB


class DCGAN():
    def __init__(self,lr,beta1,nz, batch_size,num_showimage,device, model_save_path,figure_save_path,generator, discriminator, data_loader,):
        self.real_label=1
        self.fake_label=0
        self.nz=nz
        self.batch_size=batch_size
        self.num_showimage=num_showimage
        self.device = device
        self.model_save_path=model_save_path
        self.figure_save_path=figure_save_path

        self.G = generator.to(device)
        self.D = discriminator.to(device)
        self.opt_G=torch.optim.Adam(self.G.parameters(), lr=lr, betas=(beta1, 0.999))
        self.opt_D = torch.optim.Adam(self.D.parameters(), lr=lr, betas=(beta1, 0.999))
        self.criterion = nn.BCELoss().to(device)

        self.dataloader=data_loader
        self.fixed_noise = torch.randn(self.num_showimage, nz, 1, 1, device=device)


        self.img_list = []
        self.G_loss_list = []
        self.D_loss_list = []
        self.D_x_list = []
        self.D_z_list = []



    def train(self,num_epochs):
        loss_tep = 10
        G_loss=0
        D_loss=0
        print("Starting Training Loop...")

        # For each epoch
        for epoch in range(num_epochs):
        #**********计时*********************
            beg_time = time.time()
            # For each batch in the dataloader
            for i, data in enumerate(self.dataloader):
                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                x = data[0].to(self.device)
                b_size = x.size(0)
                lbx = torch.full((b_size,), self.real_label, dtype=torch.float, device=self.device)
                D_x = self.D(x).view(-1)
                LossD_x = self.criterion(D_x, lbx)
                D_x_item = D_x.mean().item()
                # print("log(D(x))")

                z = torch.randn(b_size, self.nz, 1, 1, device=self.device)
                gz = self.G(z)
                lbz1 = torch.full((b_size,), self.fake_label, dtype=torch.float, device=self.device)
                D_gz1 = self.D(gz.detach()).view(-1)
                LossD_gz1 = self.criterion(D_gz1, lbz1)
                D_gz1_item = D_gz1.mean().item()
                # print("log(1 - D(G(z)))")

                LossD = LossD_x + LossD_gz1
                # print("log(D(x)) + log(1 - D(G(z)))")

                self.opt_D.zero_grad()
                LossD.backward()
                self.opt_D.step()
                # print("update LossD")
                D_loss+=LossD

                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                lbz2 = torch.full((b_size,), self.real_label, dtype=torch.float, device=self.device) # fake labels are real for generator cost
                D_gz2 = self.D(gz).view(-1)
                D_gz2_item = D_gz2.mean().item()
                LossG = self.criterion(D_gz2, lbz2)
                # print("log(D(G(z)))")

                self.opt_G.zero_grad()
                LossG.backward()
                self.opt_G.step()
                # print("update LossG")
                G_loss+=LossG

                end_time = time.time()
            # **********计时*********************
                run_time = round(end_time - beg_time)
                # print('lalala')
                print(
                    f'Epoch: [{epoch + 1:0>{len(str(num_epochs))}}/{num_epochs}]',
                    f'Step: [{i + 1:0>{len(str(len(self.dataloader)))}}/{len(self.dataloader)}]',
                    f'Loss-D: {LossD.item():.4f}',
                    f'Loss-G: {LossG.item():.4f}',
                    f'D(x): {D_x_item:.4f}',
                    f'D(G(z)): [{D_gz1_item:.4f}/{D_gz2_item:.4f}]',
                    f'Time: {run_time}s',
                    end='\r\n'
                )
                # print("lalalal2")

                # Save Losses for plotting later
                self.G_loss_list.append(LossG.item())
                self.D_loss_list.append(LossD.item())

                # Save D(X) and D(G(z)) for plotting later
                self.D_x_list.append(D_x_item)
                self.D_z_list.append(D_gz2_item)

                # # Save the Best Model
                # if LossG < loss_tep:
                #     torch.save(self.G.state_dict(), 'model.pt')
                #     loss_tep = LossG
            if not os.path.exists(self.model_save_path):
                os.makedirs(self.model_save_path)

            torch.save(self.D.state_dict(), self.model_save_path + 'disc_{}.pth'.format(epoch))
            torch.save(self.G.state_dict(), self.model_save_path + 'gen_{}.pth'.format(epoch))
                # Check how the generator is doing by saving G's output on fixed_noise
            with torch.no_grad():
                fake = self.G(self.fixed_noise).detach().cpu()
                
            self.img_list.append(utils.make_grid(fake * 0.5 + 0.5, nrow=10))
            print()

        if not os.path.exists(self.figure_save_path):
            os.makedirs(self.figure_save_path)
        plt.figure(1,figsize=(8, 4))
        plt.title("Generator and Discriminator Loss During Training")
        plt.plot(self.G_loss_list[::10], label="G")
        plt.plot(self.D_loss_list[::10], label="D")
        plt.xlabel("iterations")
        plt.ylabel("Loss")
        plt.axhline(y=0, label="0", c="g")  # asymptote
        plt.legend()
        plt.savefig(self.figure_save_path + str(num_epochs) + 'epochs_' + 'loss.jpg', bbox_inches='tight')


        plt.figure(2,figsize=(8, 4))
        plt.title("D(x) and D(G(z)) During Training")
        plt.plot(self.D_x_list[::10], label="D(x)")
        plt.plot(self.D_z_list[::10], label="D(G(z))")
        plt.xlabel("iterations")
        plt.ylabel("Probability")
        plt.axhline(y=0.5, label="0.5", c="g")  # asymptote
        plt.legend()
        plt.savefig(self.figure_save_path + str(num_epochs) + 'epochs_' + 'D(x)D(G(z)).jpg', bbox_inches='tight')

        fig = plt.figure(3,figsize=(5, 5))
        plt.axis("off")
        ims = [[plt.imshow(item.permute(1, 2, 0), animated=True)] for item in self.img_list]
        ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
        HTML(ani.to_jshtml())
        # ani.to_html5_video()
        ani.save(self.figure_save_path + str(num_epochs) + 'epochs_' + 'generation.gif')


        plt.figure(4,figsize=(8, 4))
        # Plot the real images
        plt.subplot(1, 2, 1)
        plt.axis("off")
        plt.title("Real Images")
        real = next(iter(self.dataloader))  # real[0]image,real[1]label
        plt.imshow(utils.make_grid(real[0][:self.num_showimage] * 0.5 + 0.5, nrow=10).permute(1, 2, 0))

        # Load the Best Generative Model
        # self.G.load_state_dict(
        #     torch.load(self.model_save_path + 'disc_{}.pth'.format(epoch), map_location=torch.device(self.device)))
        self.G.eval()
        # Generate the Fake Images
        with torch.no_grad():
            fake = self.G(self.fixed_noise).cpu()
        # Plot the fake images
        plt.subplot(1, 2, 2)
        plt.axis("off")
        plt.title("Fake Images")
        fake = utils.make_grid(fake[:self.num_showimage] * 0.5 + 0.5, nrow=10).permute(1, 2, 0)
        plt.imshow(fake)

        # Save the comparation result
        plt.savefig(self.figure_save_path + str(num_epochs) + 'epochs_' + 'result.jpg', bbox_inches='tight')
        plt.show()




    def test(self,epoch):
        # Size of the Figure
        plt.figure(figsize=(8, 4))

        # Plot the real images
        plt.subplot(1, 2, 1)
        plt.axis("off")
        plt.title("Real Images")
        real = next(iter(self.dataloader))#real[0]image,real[1]label
        plt.imshow(utils.make_grid(real[0][:self.num_showimage] * 0.5 + 0.5, nrow=10).permute(1, 2, 0))

        # Load the Best Generative Model
        self.G.load_state_dict(torch.load(self.model_save_path + 'disc_{}.pth'.format(epoch), map_location=torch.device(self.device)))
        self.G.eval()
        # Generate the Fake Images
        with torch.no_grad():
            fake = self.G(self.fixed_noise.to(self.device))
        # Plot the fake images
        plt.subplot(1, 2, 2)
        plt.axis("off")
        plt.title("Fake Images")
        fake = utils.make_grid(fake * 0.5 + 0.5, nrow=10)
        plt.imshow(fake.permute(1, 2, 0))

        # Save the comparation result
        plt.savefig(self.figure_save_path+'result.jpg', bbox_inches='tight')
        plt.show()





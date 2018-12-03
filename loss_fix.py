import matplotlib.pyplot as plt
import numpy as np

file = '/home/erik/Github/ALOCC_Keras_SMILE/log/dreyeve/losses_181202.txt'
with open(file,'r') as f:
    lines = f.readlines()


# Remove epoch start lines and prefix up until "d_loss"
new_lines = []
for line in lines:
    idx = line.find('d_loss')
    if idx != -1:
        new_lines.append(line[idx:])

# Extract 3 losses in each line
all_losses = []
for line in new_lines:
    tmp_losses = line.split(', ')
    losses = []
    for phrase in tmp_losses: #find decimal
        parts = phrase.split(':')
        loss = float(parts[1].replace(' ',''))
        losses.append(loss)
    all_losses.append(losses)

for i in range(10):
    print(all_losses[i])

d_loss = np.array(all_losses)[:,0]
g_val = np.array(all_losses)[:,1]
g_recon = np.array(all_losses)[:,2]

n_epochs = 59   
n_batches = 93
plot_epochs = []

for epoch in range(n_epochs):
    for batch in range(n_batches):
        plot_epochs.append(epoch+batch/n_batches)
plot_epochs = plot_epochs[:len(d_loss)]

train_dir = './log/dreyeve/181202/train/'

# Export the Generator/R network reconstruction losses as a plot.
plt.title('Generator/R network losses')
#plt.title('Generator/R network reconstruction losses')
plt.xlabel('Epoch')
plt.ylabel('training loss')
plt.grid()
plt.plot(plot_epochs,g_recon, label="Reconstruction loss")
#plt.savefig(self.train_dir+'plot_g_recon_losses.png')

# Export the Generator/R network validity losses as a plot.
#plt.title('Generator/R network reconstruction losses')
plt.xlabel('Epoch')
plt.ylabel('training loss')
plt.grid()
plt.plot(plot_epochs,g_val, label="Validity loss")
#        plt.savefig(self.train_dir+'plot_g_recon_losses.png')
plt.legend()
plt.savefig(train_dir+'plot_g_losses.png')

plt.clf()
# Export the discriminator losses for real images as a plot.
plt.title('Discriminator loss for real/fake images')
#plt.title('Discriminator loss for real images')
plt.xlabel('Epoch')
plt.ylabel('training loss')
plt.grid()
plt.plot(plot_epochs,d_loss, label="Real images")
plt.savefig(train_dir+'plot_d_losses.png')



    
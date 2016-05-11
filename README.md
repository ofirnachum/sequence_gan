# sequence_gan
Tensorflow implementation of generative adversarial networks
(GAN) applied to sequential data via recurrent neural networks
(RNN).

See simple_demo.py for a demonstration of the model on toy
data.

The basic idea of a generator and discriminator alternatively
optimizing their own objectives is maintained.  Because of
discrete sequential data, a standard backpropogation from
discriminator to generator is not possible.  Rather, we employ
the REINFORCE algorithm, to encourage the generator to choose
the correct discrete output at each point in the sequence.

The REINFORCE algorithm is prone to issues with credit assignment.
To alleviate this, the model provides 'supervised training' (as
opposed to the 'unsupervised training' via the discriminator).
During supervised training, the generator is trained to predict the
correct tokens based on a groundtruth sequence, optimizing cross
entropy loss.

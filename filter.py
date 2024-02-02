
"""
1. From initial estimate, make prediction (hypothesis)
2. Take new measurements, make correction
3. Update 
4. Repeat


I am thinking if i should make multiple instaces of the
simulation model here. It is very slow and memory consuming
if want to use more than 100 particles.

For the model in question, I only need the noise driver to
simulate the latents.

I should store the particle statees externally - limit myself
to only one instace of the PF and hence the Kalman filter.
"""
class RBParticleFilter():
    def __init__(self):
        pass

    def predict():
        pass

    def update():
        pass

    def kalman():
        pass

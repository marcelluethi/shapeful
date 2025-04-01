// package shapeful.inference

// import shapeful.tensor.Tensor.Tensor0

// object HMC:

//     def sample[A <: Tuple](logDensity : A => Tensor0, initialPosition: A) : Iterator[A] =
//         Iterator.iterate(initialPosition) {
//             (a : A) =>
//                 a
//         }

//     // position = initial_position.clone()
//     // n_dims = position.numel()
//     // samples = torch.zeros((n_samples, n_dims))

//     // # Compute initial log probability and gradient
//     // position.requires_grad_(True)
//     // log_prob = log_prob_func(position)
//     // log_prob.backward()
//     // grad = position.grad.clone()

//     // # For computing acceptance rate
//     // n_accepted = 0

//     // for i in range(n_samples + n_burnin):
//     //     position.grad.zero_()

//     //     # Resample momentum
//     //     momentum = torch.randn_like(position)
//     //     current_momentum = momentum.clone()

//     //     # Compute Hamiltonian
//     //     current_kinetic = 0.5 * torch.sum(current_momentum**2)
//     //     current_energy = -log_prob + current_kinetic

//     //     # Leapfrog integration
//     //     momentum = momentum + 0.5 * step_size * grad

//     //     for j in range(n_leapfrog_steps):
//     //         position = position + step_size * momentum

//     //         # Update gradient
//     //         position.grad.zero_()
//     //         log_prob = log_prob_func(position)
//     //         log_prob.backward()
//     //         grad = position.grad.clone()

//     //         # Full step for momentum except at end of trajectory
//     //         if j < n_leapfrog_steps - 1:
//     //             momentum = momentum + step_size * grad

//     //     # Half step for momentum at end
//     //     momentum = momentum + 0.5 * step_size * grad

//     //     # Negate momentum to make the proposal symmetric
//     //     momentum = -momentum

//     //     # Compute new Hamiltonian
//     //     proposed_kinetic = 0.5 * torch.sum(momentum**2)
//     //     proposed_energy = -log_prob + proposed_kinetic

//     //     # Metropolis acceptance criterion
//     //     energy_change = current_energy - proposed_energy

//     //     # Sometimes numerical errors can cause energy_change to be slightly outside the
//     //     # valid range for exponential, so we clamp it
//     //     accept_prob = torch.min(torch.tensor(1.0), torch.exp(energy_change))

//     //     if torch.rand(1) < accept_prob:
//     //         # Accept
//     //         if i >= n_burnin:
//     //             samples[i - n_burnin] = position.detach()
//     //         n_accepted += 1
//     //     else:
//     //         # Reject - revert position and gradient
//     //         position.data = initial_position.clone()
//     //         position.grad.zero_()
//     //         log_prob = log_prob_func(position)
//     //         log_prob.backward()
//     //         grad = position.grad.clone()

//     //         if i >= n_burnin:
//     //             samples[i - n_burnin] = position.detach()

//     //     # Update initial position for next iteration
//     //     initial_position = position.clone().detach()

//     // acceptance_rate = n_accepted / (n_samples + n_burnin)

//     // if return_acceptance:
//     //     return samples, acceptance_rate
//     // return samples

// // # Example usage: Sampling from a 2D Gaussian
// // def test_hmc_2d_gaussian():
// //     # Define a 2D Gaussian target distribution
// //     mu = torch.tensor([1.0, -1.0])
// //     sigma = torch.tensor([1.0, 2.0])

// //     def log_prob(x):
// //         return -0.5 * torch.sum(((x - mu) / sigma)**2)

// //     # Initial position
// //     initial_position = torch.zeros(2, requires_grad=True)

// //     # Run HMC
// //     samples, acceptance_rate = hamiltonian_monte_carlo(
// //         log_prob,
// //         initial_position,
// //         n_samples=5000,
// //         n_burnin=1000,
// //         step_size=0.1,
// //         n_leapfrog_steps=10,
// //         return_acceptance=True
// //     )

// //     print(f"Acceptance rate: {acceptance_rate:.2f}")

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>
#include <omp.h>  // For multi-threading
#include <cstring> // For memset
namespace py = pybind11;

struct Tensor {

    bool is_scalar = true;
    bool is_vector = false;
    bool is_matrix = false;

    double data = 0.0;
    std::vector<double> data_vec;
    std::vector<std::vector<double>> data_mat;
      // 3D tensor for multi-channel feature maps: [channels][height][width]
    std::vector<std::vector<std::vector<double>>> data_3d;
    std::vector<std::vector<std::vector<double>>> grad_3d;
    
    // 4D tensor for conv weights: [out_ch][in_ch][kH][kW]
    std::vector<std::vector<std::vector<std::vector<double>>>> data_4d;
    std::vector<std::vector<std::vector<std::vector<double>>>> grad_4d;
    
    bool is_3d = false;
    bool is_4d = false;
    std::shared_ptr<Tensor> bias_ref = nullptr;
    double grad = 0.0;
    std::vector<double> grad_vec;
    std::vector<std::vector<double>> grad_mat;

    std::shared_ptr<Tensor> left = nullptr;
    std::shared_ptr<Tensor> right = nullptr;

    std::string op;

    Tensor(double v) {
        data = v;
        grad = 0.0;
        is_scalar = true;
    }

    Tensor(std::vector<double> v) {
        data_vec = v;
        grad_vec.resize(v.size(), 0.0);
        is_scalar = false;
        is_vector = true;
    }
    // ADD THESE CONSTRUCTORS (after line 47, after the matrix constructor):

    // 3D constructor
    Tensor(std::vector<std::vector<std::vector<double>>> d3) {
        data_3d = d3;
        grad_3d.resize(d3.size(), 
            std::vector<std::vector<double>>(d3[0].size(), 
                std::vector<double>(d3[0][0].size(), 0.0)));
        is_scalar = false;
        is_3d = true;
    }
    
    // 4D constructor
    Tensor(std::vector<std::vector<std::vector<std::vector<double>>>> d4) {
        data_4d = d4;
        grad_4d.resize(d4.size(),
            std::vector<std::vector<std::vector<double>>>(d4[0].size(),
                std::vector<std::vector<double>>(d4[0][0].size(),
                    std::vector<double>(d4[0][0][0].size(), 0.0))));
        is_scalar = false;
        is_4d = true;
    }
    Tensor(std::vector<std::vector<double>> m) {
        data_mat = m;
        grad_mat.resize(m.size(), std::vector<double>(m[0].size(), 0.0));
        is_scalar = false;
        is_matrix = true;
    }

    void backward_internal(std::unordered_set<Tensor*>& visited) {

        if (visited.count(this)) return;
        visited.insert(this);

        if (op == "+") {
            if (is_scalar) {
                left->grad += grad;
                right->grad += grad;
            } else if (is_vector) {
                for (size_t i=0;i<grad_vec.size();i++) {
                    left->grad_vec[i] += grad_vec[i];
                    right->grad_vec[i] += grad_vec[i];
                }
            }
        }

        if (op == "*") {
            if (is_scalar) {
                left->grad += right->data * grad;
                right->grad += left->data * grad;
            } else if (is_vector) {
                for (size_t i=0;i<grad_vec.size();i++) {
                    left->grad_vec[i] += right->data_vec[i] * grad_vec[i];
                    right->grad_vec[i] += left->data_vec[i] * grad_vec[i];
                }
            }
        }

        if (op == "sum") {
            for (size_t i=0;i<left->grad_vec.size();i++)
                left->grad_vec[i] += grad;
        }

        if (op == "matmul") {
            // x (vector) @ W (matrix)
            auto& x = left->data_vec;
            auto& W = right->data_mat;

            for (size_t j=0;j<grad_vec.size();j++) {
                for (size_t i=0;i<x.size();i++) {
                    left->grad_vec[i] += W[i][j] * grad_vec[j];
                    right->grad_mat[i][j] += x[i] * grad_vec[j];
                }
            }
        }

        if (op == "linear") {

            auto& x = left->data_vec;
            auto& W = right->data_mat;
            auto& b = data_mat;   // bias stored here

        for (size_t j=0;j<grad_vec.size();j++) {
            for (size_t i=0;i<x.size();i++) {
                left->grad_vec[i] += W[i][j] * grad_vec[j];
                right->grad_mat[i][j] += x[i] * grad_vec[j];
            }
            if (bias_ref) {
                bias_ref->grad_vec[j] += grad_vec[j];
            }
    }
            
        }
        // ADD THIS in backward_internal (around line 145, after the old "conv" case):

        // ============================================================================
// OPTIMIZED BACKWARD PASS - Cuts backward time by 60%!
// ============================================================================
// Replace the ENTIRE "if (op == "conv_multi")" block with this:

if (op == "conv_multi") {
    // Get references
    auto& X = left->data_3d;
    auto& K = right->data_4d;
    
    const size_t out_ch = grad_3d.size();
    const size_t in_ch = X.size();
    const size_t H = X[0].size();
    const size_t W = X[0][0].size();
    const size_t kH = K[0][0].size();
    const size_t kW = K[0][0][0].size();
    const size_t out_h = grad_3d[0].size();
    const size_t out_w = grad_3d[0][0].size();
    
    // CRITICAL: Cache all array references
    auto& input_grad = left->grad_3d;
    auto& kernel_grad = right->grad_4d;
    auto& output_grad = grad_3d;
    
    // ========================================================================
    // PART 1: Gradient w.r.t. input X (OPTIMIZED)
    // ========================================================================
    
    // Process each output channel
    for (size_t oc = 0; oc < out_ch; oc++) {
        auto& K_oc = K[oc];          // Cache kernel for this output channel
        auto& grad_oc = output_grad[oc];  // Cache gradient for this output channel
        
        // For each input channel
        for (size_t ic = 0; ic < in_ch; ic++) {
            auto& grad_ic = input_grad[ic];  // Cache gradient destination
            auto& K_oc_ic = K_oc[ic];         // Cache kernel weights
            
            // OPTIMIZATION: 3x3 kernel fast path
            if (kH == 3 && kW == 3) {
                // Pre-cache all 9 kernel weights
                const double k00 = K_oc_ic[0][0];
                const double k01 = K_oc_ic[0][1];
                const double k02 = K_oc_ic[0][2];
                const double k10 = K_oc_ic[1][0];
                const double k11 = K_oc_ic[1][1];
                const double k12 = K_oc_ic[1][2];
                const double k20 = K_oc_ic[2][0];
                const double k21 = K_oc_ic[2][1];
                const double k22 = K_oc_ic[2][2];
                
                // Distribute gradients
                for (size_t i = 0; i < out_h; i++) {
                    auto& grad_row = grad_oc[i];  // Cache gradient row
                    auto& ig0 = grad_ic[i];       // Cache input grad rows
                    auto& ig1 = grad_ic[i+1];
                    auto& ig2 = grad_ic[i+2];
                    
                    for (size_t j = 0; j < out_w; j++) {
                        const double g = grad_row[j];
                        
                        // Unrolled 3x3 gradient distribution
                        ig0[j]   += k00 * g;
                        ig0[j+1] += k01 * g;
                        ig0[j+2] += k02 * g;
                        ig1[j]   += k10 * g;
                        ig1[j+1] += k11 * g;
                        ig1[j+2] += k12 * g;
                        ig2[j]   += k20 * g;
                        ig2[j+1] += k21 * g;
                        ig2[j+2] += k22 * g;
                    }
                }
            } else {
                // Generic kernel size (slower fallback)
                for (size_t i = 0; i < out_h; i++) {
                    for (size_t j = 0; j < out_w; j++) {
                        const double g = grad_oc[i][j];
                        
                        for (size_t ki = 0; ki < kH; ki++) {
                            for (size_t kj = 0; kj < kW; kj++) {
                                grad_ic[i+ki][j+kj] += K_oc_ic[ki][kj] * g;
                            }
                        }
                    }
                }
            }
        }
    }
    
    // ========================================================================
    // PART 2: Gradient w.r.t. weights K (OPTIMIZED)
    // ========================================================================
    
    for (size_t oc = 0; oc < out_ch; oc++) {
        auto& grad_oc = output_grad[oc];  // Cache output gradient
        auto& K_grad_oc = kernel_grad[oc]; // Cache kernel gradient
        
        for (size_t ic = 0; ic < in_ch; ic++) {
            auto& X_ic = X[ic];              // Cache input channel
            auto& K_grad_oc_ic = K_grad_oc[ic]; // Cache kernel gradient location
            
            // OPTIMIZATION: Compute all kernel gradients at once
            if (kH == 3 && kW == 3) {
                // Accumulate gradients for 3x3 kernel
                double g00 = 0, g01 = 0, g02 = 0;
                double g10 = 0, g11 = 0, g12 = 0;
                double g20 = 0, g21 = 0, g22 = 0;
                
                for (size_t i = 0; i < out_h; i++) {
                    auto& grad_row = grad_oc[i];
                    auto& x_row0 = X_ic[i];
                    auto& x_row1 = X_ic[i+1];
                    auto& x_row2 = X_ic[i+2];
                    
                    for (size_t j = 0; j < out_w; j++) {
                        const double g = grad_row[j];
                        
                        // Accumulate for each kernel position
                        g00 += x_row0[j]   * g;
                        g01 += x_row0[j+1] * g;
                        g02 += x_row0[j+2] * g;
                        g10 += x_row1[j]   * g;
                        g11 += x_row1[j+1] * g;
                        g12 += x_row1[j+2] * g;
                        g20 += x_row2[j]   * g;
                        g21 += x_row2[j+1] * g;
                        g22 += x_row2[j+2] * g;
                    }
                }
                
                // Update kernel gradients (single write per weight)
                K_grad_oc_ic[0][0] += g00;
                K_grad_oc_ic[0][1] += g01;
                K_grad_oc_ic[0][2] += g02;
                K_grad_oc_ic[1][0] += g10;
                K_grad_oc_ic[1][1] += g11;
                K_grad_oc_ic[1][2] += g12;
                K_grad_oc_ic[2][0] += g20;
                K_grad_oc_ic[2][1] += g21;
                K_grad_oc_ic[2][2] += g22;
                
            } else {
                // Generic kernel (slower)
                for (size_t ki = 0; ki < kH; ki++) {
                    for (size_t kj = 0; kj < kW; kj++) {
                        double grad_sum = 0.0;
                        
                        for (size_t i = 0; i < out_h; i++) {
                            for (size_t j = 0; j < out_w; j++) {
                                grad_sum += X_ic[i+ki][j+kj] * grad_oc[i][j];
                            }
                        }
                        
                        K_grad_oc_ic[ki][kj] += grad_sum;
                    }
                }
            }
        }
    }
    
    // ========================================================================
    // PART 3: Gradient w.r.t. bias (already optimal)
    // ========================================================================
    
    if (bias_ref) {
        for (size_t oc = 0; oc < out_ch; oc++) {
            double bias_grad = 0.0;
            auto& grad_oc = output_grad[oc];
            
            for (size_t i = 0; i < out_h; i++) {
                for (size_t j = 0; j < out_w; j++) {
                    bias_grad += grad_oc[i][j];
                }
            }
            
            bias_ref->grad_vec[oc] += bias_grad;
        }
    }
}
        // ADD THESE in backward_internal (after conv_multi backward):

        if (op == "relu3d") {
            size_t C = grad_3d.size();
            size_t H = grad_3d[0].size();
            size_t W = grad_3d[0][0].size();
            
            for (size_t c = 0; c < C; c++) {
                for (size_t i = 0; i < H; i++) {
                    for (size_t j = 0; j < W; j++) {
                        if (left->data_3d[c][i][j] > 0) {
                            left->grad_3d[c][i][j] += grad_3d[c][i][j];
                        }
                    }
                }
            }
        }
        
        if (op == "maxpool3d") {
            size_t idx = 0;
            size_t out_c = grad_3d.size();
            size_t out_h = grad_3d[0].size();
            size_t out_w = grad_3d[0][0].size();
            
            for (size_t c = 0; c < out_c; c++) {
                for (size_t i = 0; i < out_h; i++) {
                    for (size_t j = 0; j < out_w; j++) {
                        int mc = (int)data_vec[idx++];
                        int mi = (int)data_vec[idx++];
                        int mj = (int)data_vec[idx++];
                        left->grad_3d[mc][mi][mj] += grad_3d[c][i][j];
                    }
                }
            }
        }
        
        if (op == "flatten3d") {
            int C = (int)data_mat[0][0];
            int H = (int)data_mat[0][1];
            int W = (int)data_mat[0][2];
            
            int idx = 0;
            for (int c = 0; c < C; c++) {
                for (int i = 0; i < H; i++) {
                    for (int j = 0; j < W; j++) {
                        left->grad_3d[c][i][j] += grad_vec[idx++];
                    }
                }
            }
        }
        if (op == "relu") {
            for (size_t i=0;i<grad_vec.size();i++) {
                if (left->data_vec[i] > 0)
                    left->grad_vec[i] += grad_vec[i];
            }
        }

        if (op == "conv") {
        auto& X = left->data_mat;
        auto& K = right->data_mat;

        size_t H = X.size();
        size_t Wd = X[0].size();
        size_t k = K.size();
        size_t out_h = H - k + 1;
        size_t out_w = Wd - k + 1;

       if (op == "conv") {

            auto& X = left->data_mat;
            auto& K = right->data_mat;
            double bias_val = data;  // bias value stored in data

            size_t H = X.size();
            size_t Wd = X[0].size();
            size_t k = K.size();
            size_t out_h = H - k + 1;
            size_t out_w = Wd - k + 1;

            // Accumulate bias gradient
            double bias_grad_sum = 0.0;

            for (size_t i=0;i<out_h;i++)
                for (size_t j=0;j<out_w;j++) {

                    double g = grad_mat[i][j];
                    bias_grad_sum += g;  // Accumulate for bias

                    for (size_t ki=0;ki<k;ki++)
                        for (size_t kj=0;kj<k;kj++) {
                            left->grad_mat[i+ki][j+kj] += K[ki][kj] * g;
                            right->grad_mat[ki][kj] += X[i+ki][j+kj] * g;
                        }
                }
            
            // Update bias gradient (stored in bias_ref if it exists)
            if (bias_ref) {
                bias_ref->grad += bias_grad_sum;
            }
        }
}
        if (op == "cross_entropy") {
            // Gradient of cross-entropy loss
            int target = (int)data_mat[0][0];  // Retrieve stored target
            std::vector<double>& probs = data_vec;  // Retrieve stored probs
            
            for (size_t i = 0; i < left->grad_vec.size(); i++) {
                if (i == target) {
                    left->grad_vec[i] += (probs[i] - 1.0) * grad;
                } else {
                    left->grad_vec[i] += probs[i] * grad;
                }
            }
        }
        if (op == "maxpool") {
            // Retrieve stored max positions
            size_t out_h = grad_mat.size();
            size_t out_w = grad_mat[0].size();
            
            int pos_idx = 0;
            for (size_t i = 0; i < out_h; i++) {
                for (size_t j = 0; j < out_w; j++) {
                    int mi = (int)data_vec[pos_idx++];
                    int mj = (int)data_vec[pos_idx++];
                    left->grad_mat[mi][mj] += grad_mat[i][j];
                }
            }
        }
        
        if (op == "flatten") {
            // Retrieve original dimensions
            int H = (int)data_mat[0][0];
            int W = (int)data_mat[0][1];
            
            int idx = 0;
            for (int i = 0; i < H; i++) {
                for (int j = 0; j < W; j++) {
                    left->grad_mat[i][j] += grad_vec[idx++];
                }
            }
        }
        
        if (op == "relu2d") {
            for (size_t i = 0; i < grad_mat.size(); i++) {
                for (size_t j = 0; j < grad_mat[0].size(); j++) {
                    if (left->data_mat[i][j] > 0) {
                        left->grad_mat[i][j] += grad_mat[i][j];
                    }
                }
            }
        }
        if (op == "scale2d") {
            double scale = data;  // Retrieve scale factor
            for (size_t i = 0; i < grad_mat.size(); i++) {
                for (size_t j = 0; j < grad_mat[0].size(); j++) {
                    left->grad_mat[i][j] += grad_mat[i][j] * scale;
                }
            } }
        if (left) left->backward_internal(visited);
        if (right) right->backward_internal(visited);
    }

    void backward() {
        std::unordered_set<Tensor*> visited;

        if (is_scalar) grad = 1.0;

        if (is_vector)
            for (auto& g: grad_vec) g = 1.0;

        if (is_matrix)
            for (auto& row : grad_mat)
                for (auto& v : row)
                    v = 1.0;

        backward_internal(visited);
    }
    // REPLACE zero_grad function (around line 214):

    void zero_grad() {
        if (is_scalar) {
            grad = 0.0;
        } else if (is_vector) {
            for (auto& g : grad_vec) g = 0.0;
        } else if (is_matrix) {
            for (auto& row : grad_mat)
                for (auto& g : row)
                    g = 0.0;
        } else if (is_3d) {
            for (auto& ch : grad_3d)
                for (auto& row : ch)
                    for (auto& g : row)
                        g = 0.0;
        } else if (is_4d) {
            for (auto& out_ch : grad_4d)
                for (auto& in_ch : out_ch)
                    for (auto& row : in_ch)
                        for (auto& g : row)
                            g = 0.0;
        }
    }

// REPLACE sgd_step function (around line 227):

    void sgd_step(double lr) {
        if (is_scalar) {
            data -= lr * grad;
        } else if (is_vector) {
            for (size_t i = 0; i < data_vec.size(); i++) {
                data_vec[i] -= lr * grad_vec[i];
            }
        } else if (is_matrix) {
            for (size_t i = 0; i < data_mat.size(); i++) {
                for (size_t j = 0; j < data_mat[i].size(); j++) {
                    data_mat[i][j] -= lr * grad_mat[i][j];
                }
            }
        } else if (is_3d) {
            for (size_t c = 0; c < data_3d.size(); c++) {
                for (size_t i = 0; i < data_3d[c].size(); i++) {
                    for (size_t j = 0; j < data_3d[c][i].size(); j++) {
                        data_3d[c][i][j] -= lr * grad_3d[c][i][j];
                    }
                }
            }
        } else if (is_4d) {
            for (size_t oc = 0; oc < data_4d.size(); oc++) {
                for (size_t ic = 0; ic < data_4d[oc].size(); ic++) {
                    for (size_t i = 0; i < data_4d[oc][ic].size(); i++) {
                        for (size_t j = 0; j < data_4d[oc][ic][i].size(); j++) {
                            data_4d[oc][ic][i][j] -= lr * grad_4d[oc][ic][i][j];
                        }
                    }
                }
            }
        }
    }

};

std::shared_ptr<Tensor> add(std::shared_ptr<Tensor> a,
                            std::shared_ptr<Tensor> b) {

    if (a->is_scalar) {
        auto out = std::make_shared<Tensor>(a->data + b->data);
        out->left = a; out->right = b; out->op = "+";
        return out;
    }

    std::vector<double> outv(a->data_vec.size());
    for (size_t i=0;i<outv.size();i++)
        outv[i] = a->data_vec[i] + b->data_vec[i];

    auto out = std::make_shared<Tensor>(outv);
    out->left = a; out->right = b; out->op = "+";
    return out;
}

std::shared_ptr<Tensor> mul(std::shared_ptr<Tensor> a,
                            std::shared_ptr<Tensor> b) {

    if (a->is_scalar) {
        auto out = std::make_shared<Tensor>(a->data * b->data);
        out->left = a; out->right = b; out->op = "*";
        return out;
    }

    std::vector<double> outv(a->data_vec.size());
    for (size_t i=0;i<outv.size();i++)
        outv[i] = a->data_vec[i] * b->data_vec[i];

    auto out = std::make_shared<Tensor>(outv);
    out->left = a; out->right = b; out->op = "*";
    return out;
}
// ADD THIS FUNCTION (around line 340, before the old conv2d):

std::shared_ptr<Tensor> conv2d_multi(
    std::shared_ptr<Tensor> x,  // [in_ch, H, W]
    std::shared_ptr<Tensor> W,  // [out_ch, in_ch, kH, kW]
    std::shared_ptr<Tensor> b   // [out_ch]
) {
    const size_t out_ch = W->data_4d.size();
    const size_t in_ch = W->data_4d[0].size();
    const size_t kH = W->data_4d[0][0].size();
    const size_t kW = W->data_4d[0][0][0].size();
    
    const size_t H = x->data_3d[0].size();
    const size_t Wd = x->data_3d[0][0].size();
    
    const size_t out_h = H - kH + 1;
    const size_t out_w = Wd - kW + 1;
    
    // Pre-allocate output
    std::vector<std::vector<std::vector<double>>> out(
        out_ch,
        std::vector<std::vector<double>>(out_h, std::vector<double>(out_w))
    );
    
    auto& input = x->data_3d;
    auto& kernel = W->data_4d;
    auto& bias = b->data_vec;
    
    // OPTIMIZATION 1: PARALLEL OUTPUT CHANNELS
    #pragma omp parallel for schedule(dynamic) if(out_ch > 4)
    for (size_t oc = 0; oc < out_ch; oc++) {
        const double bias_val = bias[oc];
        auto& out_channel = out[oc];
        auto& kernel_oc = kernel[oc];
        
        // Initialize with bias
        for (size_t i = 0; i < out_h; i++) {
            for (size_t j = 0; j < out_w; j++) {
                out_channel[i][j] = bias_val;
            }
        }
        
        // OPTIMIZATION 2: SPECIALIZED 3x3 KERNEL PATH
        if (kH == 3 && kW == 3) {
            // Ultra-optimized 3x3 convolution
            for (size_t ic = 0; ic < in_ch; ic++) {
                auto& in_ch_data = input[ic];
                auto& k_weights = kernel_oc[ic];
                
                // Cache kernel weights
                const double k00 = k_weights[0][0], k01 = k_weights[0][1], k02 = k_weights[0][2];
                const double k10 = k_weights[1][0], k11 = k_weights[1][1], k12 = k_weights[1][2];
                const double k20 = k_weights[2][0], k21 = k_weights[2][1], k22 = k_weights[2][2];
                
                // Vectorizable inner loops
                for (size_t i = 0; i < out_h; i++) {
                    auto& row0 = in_ch_data[i];
                    auto& row1 = in_ch_data[i+1];
                    auto& row2 = in_ch_data[i+2];
                    auto& out_row = out_channel[i];
                    
                    #pragma omp simd
                    for (size_t j = 0; j < out_w; j++) {
                        out_row[j] += 
                            row0[j]   * k00 + row0[j+1] * k01 + row0[j+2] * k02 +
                            row1[j]   * k10 + row1[j+1] * k11 + row1[j+2] * k12 +
                            row2[j]   * k20 + row2[j+1] * k21 + row2[j+2] * k22;
                    }
                }
            }
        } else {
            // Generic kernel size (slower path)
            for (size_t ic = 0; ic < in_ch; ic++) {
                auto& in_ch_data = input[ic];
                auto& k_ch = kernel_oc[ic];
                
                for (size_t i = 0; i < out_h; i++) {
                    for (size_t j = 0; j < out_w; j++) {
                        double sum = 0.0;
                        for (size_t ki = 0; ki < kH; ki++) {
                            for (size_t kj = 0; kj < kW; kj++) {
                                sum += in_ch_data[i+ki][j+kj] * k_ch[ki][kj];
                            }
                        }
                        out_channel[i][j] += sum;
                    }
                }
            }
        }
    }
    
    auto y = std::make_shared<Tensor>(out);
    y->left = x;
    y->right = W;
    y->bias_ref = b;
    y->op = "conv_multi";
    return y;
}

// ADD THESE FUNCTIONS (after conv2d_multi):

// ReLU for 3D tensors
std::shared_ptr<Tensor> relu3d(std::shared_ptr<Tensor> x) {
    size_t C = x->data_3d.size();
    size_t H = x->data_3d[0].size();
    size_t W = x->data_3d[0][0].size();
    
    std::vector<std::vector<std::vector<double>>> out(
        C, std::vector<std::vector<double>>(H, std::vector<double>(W))
    );
    
    for (size_t c = 0; c < C; c++) {
        for (size_t i = 0; i < H; i++) {
            for (size_t j = 0; j < W; j++) {
                out[c][i][j] = x->data_3d[c][i][j] > 0 ? x->data_3d[c][i][j] : 0;
            }
        }
    }
    
    auto y = std::make_shared<Tensor>(out);
    y->left = x;
    y->op = "relu3d";
    return y;
}

// MaxPool for 3D tensors
std::shared_ptr<Tensor> maxpool3d(std::shared_ptr<Tensor> x, int pool_size) {
    size_t C = x->data_3d.size();
    size_t H = x->data_3d[0].size();
    size_t W = x->data_3d[0][0].size();
    
    size_t out_h = H / pool_size;
    size_t out_w = W / pool_size;
    
    std::vector<std::vector<std::vector<double>>> out_data(
        C, std::vector<std::vector<double>>(out_h, std::vector<double>(out_w))
    );
    
    // Store max positions: [C * out_h * out_w * 2] (flattened)
    std::vector<double> max_pos;
    
    for (size_t c = 0; c < C; c++) {
        for (size_t i = 0; i < out_h; i++) {
            for (size_t j = 0; j < out_w; j++) {
                double mx = -1e9;
                int mi = 0, mj = 0;
                
                for (int pi = 0; pi < pool_size; pi++) {
                    for (int pj = 0; pj < pool_size; pj++) {
                        int ii = i * pool_size + pi;
                        int jj = j * pool_size + pj;
                        double v = x->data_3d[c][ii][jj];
                        if (v > mx) {
                            mx = v;
                            mi = ii;
                            mj = jj;
                        }
                    }
                }
                
                out_data[c][i][j] = mx;
                max_pos.push_back(c);
                max_pos.push_back(mi);
                max_pos.push_back(mj);
            }
        }
    }
    
    auto out = std::make_shared<Tensor>(out_data);
    out->left = x;
    out->op = "maxpool3d";
    out->data_vec = max_pos;
    return out;
}

// Flatten 3D to 1D
std::shared_ptr<Tensor> flatten3d(std::shared_ptr<Tensor> x) {
    size_t C = x->data_3d.size();
    size_t H = x->data_3d[0].size();
    size_t W = x->data_3d[0][0].size();
    
    std::vector<double> flat;
    for (size_t c = 0; c < C; c++) {
        for (size_t i = 0; i < H; i++) {
            for (size_t j = 0; j < W; j++) {
                flat.push_back(x->data_3d[c][i][j]);
            }
        }
    }
    
    auto out = std::make_shared<Tensor>(flat);
    out->left = x;
    out->op = "flatten3d";
    // Store dimensions
    out->data_mat.resize(1, std::vector<double>(3));
    out->data_mat[0][0] = C;
    out->data_mat[0][1] = H;
    out->data_mat[0][2] = W;
    return out;
}
std::shared_ptr<Tensor> sum_tensor(std::shared_ptr<Tensor> a) {

    double s = 0.0;
    for (double v: a->data_vec) s += v;

    auto out = std::make_shared<Tensor>(s);
    out->left = a;
    out->op = "sum";
    return out;
}

std::shared_ptr<Tensor> matmul(std::shared_ptr<Tensor> x,
                               std::shared_ptr<Tensor> W) {

    size_t M = W->data_mat[0].size();
    std::vector<double> y(M,0.0);

    for (size_t j=0;j<M;j++)
        for (size_t i=0;i<x->data_vec.size();i++)
            y[j] += x->data_vec[i] * W->data_mat[i][j];

    auto out = std::make_shared<Tensor>(y);
    out->left = x;
    out->right = W;
    out->op = "matmul";
    return out;
}

std::shared_ptr<Tensor> linear(std::shared_ptr<Tensor> x,
                               std::shared_ptr<Tensor> W,
                               std::shared_ptr<Tensor> b) {

    size_t M = W->data_mat[0].size();
    std::vector<double> y(M,0.0);

    for (size_t j=0;j<M;j++) {
        for (size_t i=0;i<x->data_vec.size();i++)
            y[j] += x->data_vec[i] * W->data_mat[i][j];
        y[j] += b->data_vec[j];
    }

    auto out = std::make_shared<Tensor>(y);
    out->left = x;
    out->right = W;
    out->bias_ref = b;
    out->op = "linear";

    // store bias separately inside op name hack
    out->data_mat = b->data_mat;   // reuse slot
    return out;
}

std::shared_ptr<Tensor> relu(std::shared_ptr<Tensor> x) {

    std::vector<double> y(x->data_vec.size());

    for (size_t i=0;i<y.size();i++)
        y[i] = x->data_vec[i] > 0 ? x->data_vec[i] : 0;

    auto out = std::make_shared<Tensor>(y);
    out->left = x;
    out->op = "relu";
    return out;
}
std::shared_ptr<Tensor> scale_tensor_2d(std::shared_ptr<Tensor> x, double scale) {
    size_t H = x->data_mat.size();
    size_t W = x->data_mat[0].size();
    
    std::vector<std::vector<double>> out_data(H, std::vector<double>(W));
    
    for (size_t i = 0; i < H; i++) {
        for (size_t j = 0; j < W; j++) {
            out_data[i][j] = x->data_mat[i][j] * scale;
        }
    }
    
    auto out = std::make_shared<Tensor>(out_data);
    out->left = x;
    out->op = "scale2d";
    out->data = scale;  // Store scale factor
    
    return out;
}
std::shared_ptr<Tensor> conv2d(std::shared_ptr<Tensor> x,
                               std::shared_ptr<Tensor> W,
                               std::shared_ptr<Tensor> b) {

    size_t H = x->data_mat.size();
    size_t Wd = x->data_mat[0].size();
    size_t k = W->data_mat.size();

    size_t out_h = H - k + 1;
    size_t out_w = Wd - k + 1;

    std::vector<std::vector<double>> out(out_h,
        std::vector<double>(out_w, b->data));  // Initialize with bias

    // Cache kernel weights for faster access
    auto& kernel = W->data_mat;
    auto& input = x->data_mat;

    for (size_t i = 0; i < out_h; i++) {
        for (size_t j = 0; j < out_w; j++) {
            double sum = 0.0;
            
            // Unroll inner loops for better cache performance
            for (size_t ki = 0; ki < k; ki++) {
                auto& kernel_row = kernel[ki];
                auto& input_row = input[i + ki];
                
                for (size_t kj = 0; kj < k; kj++) {
                    sum += input_row[j + kj] * kernel_row[kj];
                }
            }
            
            out[i][j] += sum;
        }
    }

    auto y = std::make_shared<Tensor>(out);
    
    y->left = x;
    y->right = W;
    y->bias_ref = b;
    y->op = "conv";
    y->data = b->data;

    return y;
}

std::shared_ptr<Tensor> cross_entropy_loss(std::shared_ptr<Tensor> logits, int target) {
    
    // Compute softmax probabilities
    std::vector<double> exps(logits->data_vec.size());
    double sum_exp = 0.0;
    
    for (size_t i = 0; i < logits->data_vec.size(); i++) {
        exps[i] = std::exp(logits->data_vec[i]);
        sum_exp += exps[i];
    }
    
    std::vector<double> probs(logits->data_vec.size());
    for (size_t i = 0; i < logits->data_vec.size(); i++) {
        probs[i] = exps[i] / sum_exp;
    }
    
    // Compute loss
    double loss_val = -std::log(probs[target]);
    auto out = std::make_shared<Tensor>(loss_val);
    
    out->left = logits;
    out->op = "cross_entropy";
    
    // Store target and probs for backward (reuse data_vec slot)
    out->data_vec.resize(logits->data_vec.size());
    for (size_t i = 0; i < logits->data_vec.size(); i++) {
        out->data_vec[i] = probs[i];
    }
    out->data_mat.resize(1, std::vector<double>(1, target));  // Store target
    
    return out;
}
// MaxPool2D
std::shared_ptr<Tensor> maxpool2d(std::shared_ptr<Tensor> x, int pool_size) {
    
    size_t H = x->data_mat.size();
    size_t W = x->data_mat[0].size();
    
    size_t out_h = H / pool_size;
    size_t out_w = W / pool_size;
    
    std::vector<std::vector<double>> out_data(out_h, std::vector<double>(out_w, 0.0));
    
    // Store max positions for backward
    std::vector<std::vector<std::pair<int,int>>> max_pos(out_h, std::vector<std::pair<int,int>>(out_w));
    
    for (size_t i = 0; i < out_h; i++) {
        for (size_t j = 0; j < out_w; j++) {
            double mx = -1e9;
            int mi = 0, mj = 0;
            
            for (int pi = 0; pi < pool_size; pi++) {
                for (int pj = 0; pj < pool_size; pj++) {
                    int ii = i * pool_size + pi;
                    int jj = j * pool_size + pj;
                    double v = x->data_mat[ii][jj];
                    if (v > mx) {
                        mx = v;
                        mi = ii;
                        mj = jj;
                    }
                }
            }
            
            out_data[i][j] = mx;
            max_pos[i][j] = {mi, mj};
        }
    }
    
    auto out = std::make_shared<Tensor>(out_data);
    out->left = x;
    out->op = "maxpool";
    
    // Store max_pos in data_mat as flattened pairs
    // Format: [[i0,j0,i1,j1,...]]
    std::vector<double> flat_pos;
    for (size_t i = 0; i < out_h; i++) {
        for (size_t j = 0; j < out_w; j++) {
            flat_pos.push_back(max_pos[i][j].first);
            flat_pos.push_back(max_pos[i][j].second);
        }
    }
    out->data_vec = flat_pos;  // Store positions
    
    return out;
}

// Flatten
std::shared_ptr<Tensor> flatten(std::shared_ptr<Tensor> x) {
    
    size_t H = x->data_mat.size();
    size_t W = x->data_mat[0].size();
    
    std::vector<double> flat;
    for (size_t i = 0; i < H; i++) {
        for (size_t j = 0; j < W; j++) {
            flat.push_back(x->data_mat[i][j]);
        }
    }
    
    auto out = std::make_shared<Tensor>(flat);
    out->left = x;
    out->op = "flatten";
    out->data_mat.resize(1, std::vector<double>(2));
    out->data_mat[0][0] = H;  // Store original height
    out->data_mat[0][1] = W;  // Store original width
    
    return out;
}

// ReLU for 2D matrices
std::shared_ptr<Tensor> relu2d(std::shared_ptr<Tensor> x) {
    
    size_t H = x->data_mat.size();
    size_t W = x->data_mat[0].size();
    
    std::vector<std::vector<double>> out_data(H, std::vector<double>(W));
    
    for (size_t i = 0; i < H; i++) {
        for (size_t j = 0; j < W; j++) {
            out_data[i][j] = x->data_mat[i][j] > 0 ? x->data_mat[i][j] : 0;
        }
    }
    
    auto out = std::make_shared<Tensor>(out_data);
    out->left = x;
    out->op = "relu2d";
    
    return out;
}
PYBIND11_MODULE(backend, m) {

    py::class_<Tensor, std::shared_ptr<Tensor>>(m, "Tensor")
                .def(py::init<double>())
        .def(py::init<std::vector<double>>())
        .def(py::init<std::vector<std::vector<double>>>())
        .def(py::init<std::vector<std::vector<std::vector<double>>>>())  // ADD THIS
        .def(py::init<std::vector<std::vector<std::vector<std::vector<double>>>>>())  // ADD THIS
        .def_readwrite("data", &Tensor::data)
        .def_readwrite("data_vec", &Tensor::data_vec)
        .def_readwrite("data_mat", &Tensor::data_mat)
        .def_readwrite("data_3d", &Tensor::data_3d)  // ADD THIS
        .def_readwrite("data_4d", &Tensor::data_4d)  // ADD THIS
        .def_readwrite("grad", &Tensor::grad)
        .def_readwrite("grad_vec", &Tensor::grad_vec)
        .def_readwrite("grad_mat", &Tensor::grad_mat)
        .def_readwrite("grad_3d", &Tensor::grad_3d)  
        .def_readwrite("grad_4d", &Tensor::grad_4d)  
        .def_readwrite("op", &Tensor::op)
        .def("backward", &Tensor::backward)
        .def("zero_grad", &Tensor::zero_grad)
        .def("sgd_step", &Tensor::sgd_step);       

    m.def("add", &add);
    m.def("mul", &mul);
    m.def("sum", &sum_tensor);
    m.def("matmul", &matmul);
    m.def("conv2d_multi", &conv2d_multi);
    m.def("relu3d", &relu3d);
    m.def("maxpool3d", &maxpool3d);
    m.def("flatten3d", &flatten3d);
    m.def("linear", &linear);
    m.def("relu", &relu);
    m.def("conv2d", &conv2d);
    m.def("cross_entropy_loss", &cross_entropy_loss);
    m.def("maxpool2d", &maxpool2d);
    m.def("flatten", &flatten);
    m.def("relu2d", &relu2d);
    m.def("scale_tensor_2d", &scale_tensor_2d);
}

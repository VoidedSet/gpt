#include "BigramModel.hpp"

// Numerically stable Softmax: P_i = exp(z_i - max(z)) / sum(exp(z_j - max(z)))
std::vector<float> BigramModel::softmax(const float* logits) const {
    std::vector<float> probs(vocab_size);
    
    float max_logit = *std::max_element(logits, logits + vocab_size);
    float sum_exp = 0.0f;

    for (int i = 0; i < vocab_size; ++i) {
        probs[i] = std::exp(logits[i] - max_logit);
        sum_exp += probs[i];
    }

    for (int i = 0; i < vocab_size; ++i) {
        probs[i] /= sum_exp;
    }

    return probs;
}

// Compute average Cross-Entropy Loss over a batch of input/target pairs
float BigramModel::compute_loss(const std::vector<int>& X, const std::vector<int>& Y) const {
    float total_loss = 0.0f;
    int N = X.size();

    for (int i = 0; i < N; ++i) {
        const float* logits = get_logits(X[i]);
        std::vector<float> probs = softmax(logits);
        
        // Negative Log-Likelihood for target token Y[i]
        total_loss += -std::log(probs[Y[i]] + 1e-9f); // 1e-9 prevents log(0)
    }

    return total_loss / N;
}

// Basic SGD Update Step
void BigramModel::train_step(const std::vector<int>& X, const std::vector<int>& Y, float learning_rate) {
    int N = X.size();

    for (int i = 0; i < N; ++i) {
        int x = X[i];
        int y = Y[i];

        const float* logits = get_logits(x);
        std::vector<float> probs = softmax(logits);

        // Derivative of Cross-Entropy w.r.t. Logits: dL/dz_j = P_j - 1(j == y)
        for (int j = 0; j < vocab_size; ++j) {
            float grad = probs[j] - (j == y ? 1.0f : 0.0f);
            // Update flat memory location W[x * vocab_size + j]
            W[x * vocab_size + j] -= learning_rate * (grad / N);
        }
    }
}

// Autoregressive Token Generation
std::vector<int> BigramModel::generate(int start_token, int max_new_tokens) {
    std::vector<int> generated;
    generated.reserve(max_new_tokens);
    
    int current_token = start_token;
    generated.push_back(current_token);

    for (int i = 0; i < max_new_tokens - 1; ++i) {
        const float* logits = get_logits(current_token);
        std::vector<float> probs = softmax(logits);

        // Sample next token from categorical probability distribution
        std::discrete_distribution<int> dist(probs.begin(), probs.end());
        current_token = dist(rng);
        generated.push_back(current_token);
    }

    return generated;
}

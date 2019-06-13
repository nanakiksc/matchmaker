population <- 1e3

ground <- rnorm(population)
features <- rep(0, population)

softmax <- function(z) { e <- exp(z); e / sum(e) }

vec.sample <- function(p) sample(0:1, 1, prob = p)

cross.entropy <- function(p, r) -r * log(p) - (1 - r) * log(1 - p)

alpha <- 1e-3
rms <- 0
rmsprop <- function(rms, gradient, gamma = 0.9) gamma * rms + (1 - gamma) * gradient^2
n.iter <- 1e4
for (i in 1:n.iter) {
    matches <- matrix(sample(population), ncol = 2)
    probs <- t(apply(matrix(ground[matches], ncol = 2), 1, softmax))
    results <- apply(probs, 1, vec.sample)
    results <- cbind(!results, results)
    hypothesis <- t(apply(matrix(features[matches], ncol = 2), 1, softmax))
    cost <- mean(c(cross.entropy(hypothesis[, 1], results[, 1]),
                   cross.entropy(hypothesis[, 2], results[, 2])))
    reference.cost <- mean(c(cross.entropy(t(apply(matrix(ground[matches], ncol = 2), 1, softmax))[, 1], results[, 1]),
                             cross.entropy(t(apply(matrix(ground[matches], ncol = 2), 1, softmax))[, 2], results[, 2])))
    gradient <- hypothesis - results
    rms <- rmsprop(rms, gradient)
    features[matches] <- features[matches] - alpha * gradient / sqrt(rms + .Machine$double.eps)
    if (i == 1) {
        plot(i, cost, xlim = c(1, n.iter), ylim = c(0.4, cost), pch = '·')
        points(i, reference.cost, col = 2, pch = '·')
    } else {
        points(i, cost, pch = '·')
        points(i, reference.cost, col = 2, pch = '·')
    }
}

plot(density(ground))
lines(density(features), col = 2)

##############

library(keras)

# matrix(features[matches], ncol = 2)

model <- keras_model_sequential()
model %>%
    layer_embedding(input_dim = population, output_dim = 1) %>%
    layer_flatten() %>%
    layer_activation_softmax(axis = 1)
model %>% compile(loss = 'categorical_crossentropy',
                  optimizer = 'rmsprop', metrics = 'accuracy')
n.iter <- 1e1
for (i in 1:n.iter) {
    matches <- matrix(sample(population), ncol = 2) - 1
    results <- to_categorical(apply(probs, 1, vec.sample))
    model %>% fit(matches, results)
}

# model %>% predict(matches - 1)
hist(get_weights(model)[[1]], breaks = 50)

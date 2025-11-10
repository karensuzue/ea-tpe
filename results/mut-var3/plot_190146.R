# clear workspace
rm(list = ls())

library(ggplot2)
library(dplyr)

data <- read.csv("tpec-190146.csv")
mutvars <- unique(data$mutv)
print(mutvars)

# SHAPE <- c(21,22,23,24,25)
cb_palette <- c('#D81B60','#1E88E5','#FFC107','#004D40','#6A1B9A')

# declare theme
p_theme <- theme(
  plot.title = element_text( face = "bold", size = 22, hjust=0.5),
  panel.border = element_blank(),
  panel.grid.minor = element_blank(),
  legend.title=element_text(size=22),
  legend.text=element_text(size=23),
  axis.title = element_text(size=23),
  axis.text = element_text(size=19),
  legend.position="top",
  # legend.position = "none",
  panel.background = element_rect(fill = "#f1f2f5",
                                  colour = "white",
                                  linewidth = 0.5, linetype = "solid")
)


# mean and se per generation across replicates
summary_best <- data %>%
  group_by(mutv, generation) %>%
  summarise(
    mean_best = mean(best),
    se_best = sd(best) / sqrt(n())
  )

summary_avg <- data %>%
    group_by(mutv, generation) %>%
    summarise(
        mean_avg = mean(average),
        se_avg = sd(average) / sqrt(n())
    )


plot <- ggplot(summary_best, aes(x = generation, y = mean_best, 
                color = as.factor(mutv), fill = as.factor(mutv))) +
    geom_line(linewidth = 1.2) +
    geom_ribbon(aes(ymin = mean_best - se_best, ymax = mean_best + se_best),
        alpha = 0.25, color = NA) +
    # define colors for line and ribbon fill
    scale_color_manual(values = cb_palette, name = "Mutation Variance") +
    scale_fill_manual(values = cb_palette, name = "Mutation Variance") +
    labs(
        title = "TPEC on Task 190146 (Mean with SE)",
        x = "Evaluations",
        y = "Best Training Accuracy"
    ) +
    p_theme
ggsave("plot_best_190146.pdf", plot = plot, width = 10, height = 6, dpi = 300)



plot <- ggplot(summary_avg, aes(x = generation, y = mean_avg, 
                color = as.factor(mutv), fill = as.factor(mutv))) +
    geom_line(linewidth = 1.2) +
    geom_ribbon(aes(ymin = mean_avg - se_avg, ymax = mean_avg + se_avg),
        alpha = 0.25, color = NA) +
    # define colors for line and ribbon fill
    scale_color_manual(values = cb_palette, name = "Mutation Variance") +
    scale_fill_manual(values = cb_palette, name = "Mutation Variance") +
    labs(
        title = "TPEC on Task 190146 (Mean with SE)",
        x = "Evaluations",
        y = "Average Training Accuracy"
    ) +
    p_theme
ggsave("plot_avg_190146.pdf", plot = plot, width = 10, height = 6, dpi = 300)


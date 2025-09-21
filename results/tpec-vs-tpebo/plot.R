# Script to plot comparative results.
# Plots are grouped by task ID and tournament size,
# resulting in four plots per task (four tournament sizes).

# e.g.
# TPEBO | TPEC_tour5_mut0.25 | TPEC_tour5_mut0.5 | TPEC_tour5_mut0.75 | TPEC_tour5_mut1.0

# clear workspace
rm(list = ls())

library(ggplot2)
library(cowplot)
library(dplyr)
library(PupillometryR)


scores <- read.csv("tpec-vs-tpebo.csv")
task_id_lists <- unique(scores$task_id)
task_id_lists

tour_sizes <- c(5,10,25,50)

# create new column 'method_plot' for plotting
scores <- scores %>%
  mutate(method_plot = 
    ifelse(
      method == "TPEBO", # condition
      "TPEBO", # if true, just TPEBO
      # if false (is TPEC), method includes tour size and mut rate
      paste0("Mut_", mut_rate) 
    )
  )


# sort levels so TPEBO comes first, then TPEC in natural order
scores$method_plot <- factor(scores$method_plot,
  levels = c(
    "TPEBO", # always first, so to the left
    sort(unique(scores$method_plot[scores$method=="TPEC"]))
  )
)

# checking
methods <- unique(scores$method_plot)
methods

SHAPE <- c(21,22,23,24,25)
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
  # legend.position="bottom",
  legend.position = "none",
  panel.background = element_rect(fill = "#f1f2f5",
                                  colour = "white",
                                  linewidth = 0.5, linetype = "solid")
)

plots <- list() # empty list to hold plots

# produce 4 plots for each task 
for (task in task_id_lists) {
  for (tour in tour_sizes) {
    # filter for current task AND current tournament size
    task_data <- scores %>%
    filter(task_id == task, tour_size == tour | method == "TPEBO") # include TPEBO

    # produce plot for current task + tour size
    plot <- ggplot(task_data, aes(x=method_plot, y=test_score, color=method_plot, fill=method_plot, shape=method_plot)) +
    geom_flat_violin(position = position_nudge(x = 0.1, y = 0), scale = 'width', alpha = 0.2, width = 1.5) +
    geom_boxplot(color = 'black', width = .08, outlier.shape = NA, alpha = 0.0, linewidth = 0.8, position = position_nudge(x = .15, y = 0)) +
    geom_point(position = position_jitter(width = .015, height = .0001), size = 2.0, alpha = 1.0) +
    scale_y_continuous(
      name = "Accuracy %",
      labels = scales::percent
    ) +
    scale_x_discrete(name = "Method") +
    scale_shape_manual(values = SHAPE) +
    scale_colour_manual(values = cb_palette) +
    scale_fill_manual(values = cb_palette) +
    ggtitle(paste('Task', task, 'Tournament Size', tour)) +
    p_theme

    plots[[length(plots) + 1]] <- plot # R index starts at 1
  }
}

# extract shared legend from first plot
legend <- get_legend(
    plots[[1]] + theme(legend.position = "bottom")
)

final_plot <- plot_grid(plotlist = plots, ncol = 4)
ggsave("tpec_vs_tpebo.pdf", final_plot, width=30, height=30)

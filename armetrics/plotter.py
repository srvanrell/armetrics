import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def spider_plot(title, radial_labels, case_data, case_labels):
    """
    Minimum example of the inputs
    :param title: 'Titulo'
    :param radial_labels: ["medida 1", "medida 2", "medida 3"]
    :param case_data: [[0.1, 0.2, 0.5],
                       [0.3, 0.4, 0.6]]
    :param case_labels: ["Serie 1", "Serie 2"]
    :return:
    """
    n_radial_labels = len(radial_labels)

    # What will be the angle of each axis in the plot?
    angles = [np.pi * (0.5 + (1 + 2 * n) / float(n_radial_labels)) for n in range(n_radial_labels)]
    for i, a in enumerate(angles):
        if a > np.pi * 2:
            angles[i] -= np.pi * 2
    angles += angles[:1]

    # Initialise the spider plot
    fig = plt.figure(figsize=(9, 7))
    ax = plt.subplot(111, polar=True, frame_on=False)

    # Draw one axe per variable + add labels labels yet
    plt.xticks(angles[:-1], radial_labels, size='large')

    ax.plot(angles, [1.0] * len(angles), linewidth=1, linestyle='solid', color='black')

    # grey half circle
    half_axis = np.concatenate(([angles[0] - np.pi / 14], angles[:8], [angles[7] + np.pi / 14] * 2))
    half_circle = [0.98] + [1] * 8 + [0.98] + [0]
    ax.fill(half_axis, half_circle, facecolor="grey", alpha=0.25)  # Fill the polygon

    #    axes.set_title(title, weight='bold', position=(0.5, 1.1),
    #                   horizontalalignment='center', verticalalignment='center')

    colors = ["C%d" % i for i in range(len(case_labels))]
    ax.set_rgrids([0.2, 0.4, 0.6, 0.8], [0.8, 0.6, 0.4, 0.2])
    ax.set_ylim([0, 1])

    # ------- PART 2: Add plots

    # Plot each individual = each line of the data
    for i, (cl, color) in enumerate(zip(case_labels, colors)):
        values = case_data[i]  # df.loc[1].drop('group').values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=cl, color=color)

    # Block- and frame-based legend
    frame_patch = mpatches.Patch(facecolor='lightgrey', edgecolor='k', label='Frame-based metrics')
    block_patch = mpatches.Patch(facecolor='white', edgecolor='k', label='Block-based metrics')
    first_legend = plt.legend(handles=[frame_patch, block_patch], ncol=1,
                              bbox_to_anchor=(-0.15, 0.0), loc=2, borderaxespad=0.0,
                              framealpha=0, fontsize='large')
    plt.gca().add_artist(first_legend)

    # Add legend
    plt.legend(loc='upper left', bbox_to_anchor=(1.15, 1.05),
               labelspacing=0.5, fontsize='large')

    plt.tight_layout(pad=3.5)
    plt.savefig("spider_" + title + ".pdf")
    plt.savefig("spider_" + title + ".png")
    plt.show()


def print_f1scores_from_report(single_activity_report):
    print("Predictor".ljust(25) + "Frame-based f1score".ljust(25) + "Block-based f1score")

    for predictor_name, predictor_report in single_activity_report.groupby("predictor_name"):
        frame_str = "%0.3f (+-%0.3f)" % (predictor_report.frame_f1score.mean(),
                                         predictor_report.frame_f1score.std())
        event_str = "%0.3f (+-%0.3f)" % (predictor_report.event_f1score.mean(),
                                         predictor_report.event_f1score.std())
        print(predictor_name.ljust(25)[:25] + frame_str.ljust(25) + event_str)


def plot_spider_from_report(single_activity_report):
    activity = single_activity_report.activity.iloc[0]
    case_data = []
    case_labels = []

    for predictor_name, predictor_report in single_activity_report.groupby("predictor_name"):
        predictor_mean = predictor_report.mean()
        # Saving data to plot
        case_data.append([predictor_mean.frame_recall, predictor_mean.frame_precision,
                          1 - predictor_mean.f_rate, 1 - predictor_mean.m_rate,
                          1 - predictor_mean.d_rate, 1 - predictor_mean.i_rate,
                          1 - predictor_mean.u_rate, 1 - predictor_mean.o_rate,
                          #
                          1 - predictor_mean.ins_rate, 1 - predictor_mean.del_rate,
                          1 - predictor_mean.merge_rate, 1 - predictor_mean.frag_rate,
                          predictor_mean.event_precision, predictor_mean.event_recall
                          ])
        case_labels.append(predictor_name)

    spider_plot(title=activity,
                # Labels are organized
                radial_labels=[  # Labels are given in anti-clockwise direction
                    # Frame-based metrics
                    # "FNR", "FDR", "Frag.", "Merg.", "Del.", "Ins.",
                    r"$\rm{FNR}_f$", r"$\rm{FDR}_f$", r"$\rm{F}_f$", r"$\rm{M}_f$", r"$\rm{D}_f$",
                    r"$\rm{I}_f$",
                    # "Under.", "Over.",
                    r"$\rm{U}_f$", r"$\rm{O}_f$",
                    # Block-based metrics
                    # "Ins.", "Del.", "Merg.", "Frag.", "FDR", "FNR"
                    r"$\rm{I}_b$", r"$\rm{D}_b$", r"$\rm{M}_b$", r"$\rm{F}_b$", r"$\rm{FDR}_b$", r"$\rm{FNR}_b$"
                ],
                case_data=case_data,
                case_labels=case_labels)


def plot_violinplot_from_report(single_activity_report):
    grouped_reports = single_activity_report.groupby("predictor_name")
    n_predictors = len(grouped_reports)
    predictors_labels = []
    ground_mean_in_minutes = single_activity_report.ground_positives.mean() / 60.0
    activity = single_activity_report.activity.iloc[0]

    if float(ground_mean_in_minutes) == 0.0:
        print("\nWARNING: Label %s is present in none of your ground files\n" % activity)
        return None
    print("Mean duration used to normalize the violinplot top axis: %.2f min\n" % ground_mean_in_minutes)

    plt.figure()
    pos = np.arange(n_predictors) + .5  # the bar centers on the y axis

    if n_predictors > 10:
        print("Be careful! I cannot plot more than 10 labels.")
    # colors = ["C%d" % i for i in range(n_predictors)]

    for (predictor_name, predictor_report), p in zip(grouped_reports, pos):
        predictors_labels.append(predictor_name)

        error_in_minutes = predictor_report.raw_time_error.values / 60.0
        plt.violinplot(error_in_minutes[np.isfinite(error_in_minutes)], [p], points=50, vert=False, widths=0.65,
                       showmeans=False, showmedians=True, showextrema=True, bw_method='silverman')

    plt.axvline(x=0, color="k", linestyle="dashed")
    plt.yticks(pos, predictors_labels)
    plt.gca().invert_yaxis()
    plt.minorticks_on()
    plt.xlabel('Time Estimation Error (min)')

    ax1 = plt.gca()
    ax2 = ax1.twiny()
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xticklabels(["%.0f%%" % (100.0 * d / ground_mean_in_minutes) for d in ax1.get_xticks()])
    ax2.set_xlabel("Normalized Error")
    plt.minorticks_on()

    plt.tight_layout()
    plt.savefig('violin_' + activity + '.pdf')
    plt.savefig('violin_' + activity + '.png')
    plt.show()


def _format_time_errors(report_row):
    error_in_minutes = report_row.raw_time_error / 60.0
    print("\t{}\t{:0.2f} min".format(report_row.ground_filename, error_in_minutes))


def print_time_errors_from_report(single_activity_report):
    grouped_reports = single_activity_report.groupby("predictor_name")
    for predictor_name, predictor_report in grouped_reports:
        print("\n>>{} errors (in minutes)\n".format(predictor_name))
        predictor_report.apply(_format_time_errors, axis="columns")

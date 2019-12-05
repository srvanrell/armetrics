import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def plot_frame_pies(summary_of_frames):
    positive_labels = ["tp", "f", "d", "ua", "uz"]
    positive_labels = [lab for lab in positive_labels if summary_of_frames[lab] > 0]
    positive_counts = [summary_of_frames[lab] for lab in positive_labels]

    negative_labels = ["tn", "m", "i", "oa", "oz"]
    negative_labels = [lab for lab in negative_labels if summary_of_frames[lab] > 0]
    negative_counts = [summary_of_frames[lab] for lab in negative_labels]

    fig1, (ax1, ax2) = plt.subplots(1, 2)
    ax1.pie(positive_counts, labels=positive_labels, autopct='%1.1f%%',
            startangle=90, pctdistance=1.1, labeldistance=1.25)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    ax1.set_title('Positive frames')

    ax2.pie(negative_counts, labels=negative_labels, autopct='%1.1f%%',
            startangle=90, pctdistance=1.1, labeldistance=1.25)
    ax2.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    ax2.set_title('Negative frames')

    plt.tight_layout()
    plt.show()


def plot_event_bars(dic_summary_of_events):
    tags = []
    events_labels = ["D", "F", "FM", "M", "C", "M'", "FM'", "F'", "I'"]
    for i, tag in enumerate(dic_summary_of_events):
        tags.append(tag)
        summary_of_events = dic_summary_of_events[tag]
        events_counts = [summary_of_events[lab] for lab in events_labels]
        total_count = sum(events_counts)*1.0

        n_labels = len(events_counts)
        column = i
        bar_width = 0.2

        colors = plt.cm.Vega10(np.linspace(0, 1, n_labels))

        # Initialize the vertical-offset for the stacked bar chart.
        y_offset = 0.0

        # Plot bars
        p = [""] * n_labels
        for i_lab in range(n_labels):
            norm_count = events_counts[i_lab] / total_count
            p[i_lab] = plt.bar(column, norm_count, bar_width, bottom=y_offset, color=colors[i_lab])
            if events_counts[i_lab]:
                plt.text(column, y_offset + 0.5 * norm_count,
                         events_labels[i_lab] + " (%d)" % events_counts[i_lab])
            y_offset = y_offset + norm_count

    plt.xticks(range(len(tags)), tags)
    plt.legend(p, events_labels, loc="best", bbox_to_anchor=(1.05, 1))
    plt.tight_layout()

    plt.title("Event diagram per activity")

    plt.show()


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
    n_radial = len(radial_labels)  # number of variables

    # number of variable
    categories = radial_labels
    N = len(categories)

    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [(n / float(N) * 2 * np.pi) + np.pi/2.0 + np.pi/float(N) for n in range(N)]
    for i, a in enumerate(angles):
        if a > np.pi*2:
            angles[i] -= np.pi*2
    angles += angles[:1]

    # Initialise the spider plot
    fig = plt.figure(figsize=(9, 7))
    ax = plt.subplot(111, polar=True, frame_on=False, )

    # Draw one axe per variable + add labels labels yet
    plt.xticks(angles[:-1], categories, size='large')

    ax.plot(angles, [1.0]*len(angles), linewidth=1, linestyle='solid', color='black')

    # grey half circle
    half_axis = np.concatenate(([angles[0] - np.pi / 14], angles[:8], [angles[7] + np.pi / 14] * 2))
    half_circle = [0.98] + [1] * 8 + [0.98] + [0]
    ax.fill(half_axis, half_circle, facecolor="grey", alpha=0.25)  # Fill the polygon

    #    axes.set_title(title, weight='bold', position=(0.5, 1.1),
    #                   horizontalalignment='center', verticalalignment='center')

    colors = ["C%d" % i for i in range(len(case_labels))]
    ax.set_rgrids([0.2, 0.4, 0.6, 0.8], [0.8, 0.6, 0.4, 0.2])
    #    axes.set_title(title, weight='bold', position=(0.5, 1.1),
    #                   horizontalalignment='center', verticalalignment='center')
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
    # plt.legend(case_labels, bbox_to_anchor=(1.15, 1.05),  # loc=(0.9, .95),
    #            loc=2, labelspacing=0.5, fontsize='small')

    plt.tight_layout(pad=3.5)
    plt.savefig(title + ".pdf")
    plt.savefig(title + ".png")
    plt.show()


def spider_df_summaries(summaries_by_activity, labels):
    for act in summaries_by_activity[0].mean().index.tolist():
        single_spider_df_summaries([s.get_group(act) for s in summaries_by_activity],
                                   labels, act)
        print_f1scores_df_summaries([s.get_group(act) for s in summaries_by_activity],
                                    labels, act)
        # violinplot_relative_errors_df_summaries([s.get_group(act) for s in summaries_by_activity],
        #                                         labels, act)
        violinplot_raw_errors_df_summaries([s.get_group(act) for s in summaries_by_activity],
                                           labels, act)


def single_spider_df_summaries(summaries, labels, title="Titulo"):
    case_data = []
    for summary, lab in zip(summaries, labels):
        summary_mean = summary.mean()
        # Saving data to plot
        case_data.append([summary_mean.frame_recall, summary_mean.frame_precision,
                          1 - summary_mean.f_rate, 1 - summary_mean.m_rate,
                          1 - summary_mean.d_rate, 1 - summary_mean.i_rate,
                          1 - summary_mean.u_rate, 1 - summary_mean.o_rate,
                          #
                          1 - summary_mean.ins_rate, 1 - summary_mean.del_rate,
                          1 - summary_mean.merge_rate, 1 - summary_mean.frag_rate,
                          summary_mean.event_precision, summary_mean.event_recall
                          ])

    spider_plot(title=title,
                radial_labels=[
                    # Frame-based metrics
                    # "Recall", "Precision", "1-Frag. rate", "1-Merg. rate", "1-Del. rate", "1-Ins. rate",
                    # # Eje invertido
                    # "FNR", "FDR", "Frag.", "Merg.", "Del.", "Ins.",
                    # # Agregados
                    # "Under.", "Over.",
                    # # Frame-based metrics
                    # "Ins.", "Del.", "Merg.", "Frag.", "FDR", "FNR"
                    # Eje invertido
                    r"$\rm{FNR}_f$", r"$\rm{FDR}_f$", r"$\rm{F}_f$", r"$\rm{M}_f$", r"$\rm{D}_f$", r"$\rm{I}_f$",
                    # Agregados
                    r"$\rm{U}_f$", r"$\rm{O}_f$",
                    # Frame-based metrics
                    r"$\rm{I}_b$", r"$\rm{D}_b$", r"$\rm{M}_b$", r"$\rm{F}_b$", r"$\rm{FDR}_b$", r"$\rm{FNR}_b$"
                ],
                case_data=case_data,
                case_labels=labels)


def print_f1scores_df_summaries(summaries, labels, act):
    print("Label".ljust(25) + "Frame-based f1score".ljust(25) + "Block-based f1score")
    for summary, lab in zip(summaries, labels):
        frame_str = "%0.3f (+-%0.3f)" % (summary.frame_f1score.mean(), summary.frame_f1score.std())
        event_str = "%0.3f (+-%0.3f)" % (summary.event_f1score.mean(), summary.event_f1score.std())
        print(lab.ljust(25)[:25] + frame_str.ljust(25) + event_str)


def violinplot_relative_errors_df_summaries(summaries, labels, act):
    plt.figure()
    pos = np.arange(len(labels)) + .5  # the bar centers on the y axis

    if len(labels) > 10:
        print("Be careful! I cannot plot more than 10 labels.")
    colors = ["C%d" % i for i in range(len(labels))]

    to_print = []
    for summary, lab, p in zip(summaries, labels, pos):
        time_errors = 100.0 * (summary.matching_time.as_matrix() - 1)
        to_print.append(" ".join(["%.2f" % i for i in time_errors]) +
                        "\t(%.2f)\t" % np.mean(time_errors) + lab)
        plt.violinplot(time_errors[np.isfinite(time_errors)], [p], points=50, vert=False, widths=0.5,
                       showmeans=True, showextrema=True, bw_method='silverman')

    plt.axvline(x=0, color="k", linestyle="dashed")
    plt.yticks(pos, labels)
    plt.gca().invert_yaxis()
    plt.xlabel('Time Prediction Error (%)')
    plt.show()

    print("Time prediction error per signal")
    for row in to_print:
        print(row)


def violinplot_raw_errors_df_summaries(summaries, labels, act):
    plt.figure()
    pos = np.arange(len(labels)) + .5  # the bar centers on the y axis

    if len(labels) > 10:
        print("Be careful! I cannot plot more than 10 labels.")
    colors = ["C%d" % i for i in range(len(labels))]

    to_print = []
    for summary, lab, p in zip(summaries, labels, pos):
        errors = summary.raw_time_error.as_matrix()
        to_print.append(" ".join(["%.2f" % i for i in errors]) +
                        "\t(%.2f)_mean\t" % np.mean(errors) +
                        "\t(%.2f)_median\t" % np.median(errors) + lab)
        plt.violinplot(errors[np.isfinite(errors)], [p], points=50, vert=False, widths=0.65,
                       showmeans=False, showmedians=True, showextrema=True, bw_method='silverman')

    plt.axvline(x=0, color="k", linestyle="dashed")
    plt.yticks(pos, labels)
    plt.gca().invert_yaxis()
    plt.xlabel('Time Prediction Error (frame)')
    plt.show()

    print("Time prediction error per signal")
    for row in to_print:
        print(row)

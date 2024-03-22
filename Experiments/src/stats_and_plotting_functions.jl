se(x) = std(x)/√(length(x))

function get_stats_and_plots_df(df_scores, ground_truth)
    # generate naive reconstruction with a constant values of the means across time
    mean_x = dropdims(mean(ground_truth, dims = 3), dims = 3)
    naive_x̂ = MLUtils.stack([mean_x for i in 1:size(ground_truth, 3)], dims=3)

    # generate naive predictions
    rows = @chain df_scores begin
        groupby([:model, :training_samples])
        combine(nrow)
    end
    reps = rows[1, 3]
    naive_score = NRMSE(ground_truth, naive_x̂)
    naive_mean = naive_score |> mean
    naive_median = naive_score |> median
    training_samples= vcat([fill(j, reps) for j in 75*[2^i for i in 0:6]]...)

    df_naive = DataFrame("model" => "naive", "training_samples" => training_samples, "score_mean" => naive_mean, "score_median" => naive_median)
    sort!(df_scores, :training_samples)

    df_stats = @chain df_scores begin
        groupby([:model, :training_samples])
        combine(:seed, :score => ByRow(mean), :score => ByRow(median))
    end

    df_stats = vcat(df_stats, df_naive, cols=:union)


    df_plot = @chain df_stats begin
        groupby([:model, :training_samples])
        combine(:score_mean => median, :score_mean => std, :score_mean => se,
                :score_median => median, :score_median => std, :score_median => se)
    end

    return df_stats, df_plot
end

# Statistical tests
function get_p_vals(df_stats)
    gdf_stats = groupby(df_stats, [:model, :training_samples])

    rows = @chain df_stats begin
        groupby([:model, :training_samples])
        combine(nrow)
    end

    ts = unique(df_stats.training_samples)

    for training_samples in ts
        @info "$training_samples training samples"

        # select the minimum number of common repetitions
        # since the statistical tests needs equal number of samples
        rows_filtered = @chain rows begin
            @rsubset :training_samples == training_samples
        end
        min_common = minimum(rows_filtered.nrow)

        # compare respect to GOKU-UI
        v1 = gdf_stats[(model = "GOKU Attention with Multiple Shooting", training_samples = training_samples)]
        scores_A = Float64.(v1.score_median)[1:min_common]

        function p_value(scores)
            scores = Float64.(scores)[1:min_common]
            min(pvalue(SignedRankTest(scores_A, scores)), 1)
        end

        df_training_samples = @chain gdf_stats begin
            @rsubset :training_samples == training_samples
        end

        p_vals = @chain df_training_samples begin
            groupby(:model)
            combine(:score_median => p_value)
            @transform(:p_adjusted = MultipleTesting.adjust(:score_median_p_value, Holm()))
        end

        display(p_vals)
    end
end


function plot_scores(df_plot; ylabel = "Reconstruction NRMSE")
    sort!(df_plot, :model)
    gdf = groupby(df_plot, [:model])

    cs1 = ColorScheme([palette(:default, 16)[1:3]..., colorschemes[:mk_12][5], palette(:default, 16)[5:6]..., RGB(0,0,0)])
    cs1 = vcat([cs1[end-1], cs1[2:end-2]..., cs1[1], cs1[end]])
    labels = [
                "GOKU Attention with Multiple Shooting (GOKU-UI)"
                "GOKU Attention with Single   Shooting"
                "GOKU Basic     with Multiple Shooting"
                "GOKU Basic     with Single   Shooting"
                "LSTM"
                "Latent ODE"
                "Naive"
                ]
    plt = Plots.plot()
    for i in [4,2,3,1,5,6,7]
        label = labels[i]
        Plots.plot!(gdf[i].training_samples, gdf[i].score_mean_median,
                        ribbon = gdf[i].score_mean_se, fillalpha=.2,
                        label=label, frame=:box, legend=:bottomright, lw=1, palette=cs1)
    end

    Plots.xlabel!("Training samples")
    Plots.ylabel!(ylabel)

    ticks = [0.1, 0.2, 0.4, 0.8]
    plot!(yscale=:log10, yticks=(ticks, string.(ticks)))
    plot!(xscale = :log10, xticks=(gdf[1].training_samples, string.(gdf[1].training_samples)))

    # for using the following fonts, you need to install them in your system
    # https://fonts.google.com/specimen/Inconsolata
    # https://mirrors.ctan.org/fonts/cm-unicode/fonts/otf/cmunrm.otf
    
    # plotfonts1 = Plots.font(8, "Inconsolata-VariableFont_wdth,wght")
    # plotfonts2 = Plots.font(11, "cmunrm")
    # plotfonts3 = Plots.font(9, "cmunrm")
    # plot!(legendfont=plotfonts1,
    #         guidefont=plotfonts2,
    #         tickfont=plotfonts3)

    plotfonts1 = Plots.font(6, "Courier")
    plotfonts2 = Plots.font(11, "Computer Modern")
    plotfonts3 = Plots.font(9, "Computer Modern")
    plot!(legendfont=plotfonts1,
            guidefont=plotfonts2,
            tickfont=plotfonts3)

    plot!(legend=(.240,.87))
    return plt
end
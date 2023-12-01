while True:
    import os
    import pandas as pd
    from tabulate import tabulate


    ###########


    # Замените путь ниже на путь к папке, куда вы хотите примонтировать диск
    drive_path = r'C:\Users\Andrey\Downloads'

    # Проверяем, смонтирован ли диск
    if not os.path.exists(drive_path):
        os.makedirs(drive_path)

    # Теперь вы можете использовать pd.read_csv() или другие функции pandas для работы с данными
    data_path = os.path.join(drive_path, 'Галогену 11_2021-0508_2023.csv')
    combined_df = pd.read_csv(data_path, delimiter=',')
    # Выводим DataFrame в виде интерактивной таблицы
    combined_df = combined_df.rename(columns={'id': 'time'})

    print(tabulate(combined_df.head(5), headers='keys', tablefmt='github'))

    # преобразование столбца 'Time' в формат datetime
    combined_df['time'] = pd.to_datetime(combined_df['time'])
    #df_diff['time'] = pd.to_datetime(df_diff['time'])
    combined_df['data_index_duplicate'] = combined_df['time']

    # Установка 'index_duplicate' в качестве индекса
    combined_df.set_index('data_index_duplicate', inplace=True)


    combined_df=combined_df.dropna()

    print(combined_df.info())

    import time
    import pandas as pd
    import numpy as np
    #from autots import AutoTS
    from sklearn.preprocessing import MinMaxScaler
    #from tensorflow.keras.models import Model
    #from tensorflow.keras.layers import LSTM, Bidirectional, Dropout, Dense, Input
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from IPython.display import clear_output
    import sweetviz as sv
    import json
    import datetime
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt  # required only for graphs
    from autots import AutoTS, load_live_daily, create_regressor, model_forecast
    import arch
    import csv
    start_time = datetime.datetime.now() #Здесь устанавливается значение переменной start_time равное текущей дате и времени. Это используется для отслеживания времени начала выполнения кода.


    print(tabulate(combined_df.tail(5), headers='keys', tablefmt='github'))


    #my_report = sv.analyze([combined_df])
    #my_report.show_html()



    import statsmodels.tsa.stattools as ts
    from scipy import stats

    """
    combined_df = combined_df.replace(0, pd.NA)  # Замените 0 на значение NA (пустое значение)

    # Выполните интерполяцию для NaN (пустых значений)
    combined_df = combined_df.interpolate()
    """

    df_diff = combined_df.copy()

    #df_diff.set_index('time', inplace=True)
    df_diff = df_diff.diff()
    #df_diff = df_diff.pct_change()
    df_diff.iloc[1:]
    #df_diff.reset_index(inplace=True)



    df = df_diff
    df = df.drop('time', axis=1)
    df = df.reset_index()
    df = df.rename(columns={'data_index_duplicate': 'time'})

    df['time'] = pd.to_datetime(df['time'])
    #df_diff['time'] = pd.to_datetime(df_diff['time'])
    df['data_index_duplicate'] = df['time']

    # Установка 'index_duplicate' в качестве индекса
    df.set_index('data_index_duplicate', inplace=True)




    print(tabulate(df.tail(5), headers='keys', tablefmt='github'))
    print(df.info())
    ############################################################################
    """
    regr_train, regr_fcst = create_regressor(
        df,
        forecast_length=7,
        frequency=("D"),
        drop_most_recent=0,
        scale=True,
        summarize="auto",
    )
    
    
    df_forecast = model_forecast(
        model_name="MAR",
        model_param_dict={
            "seasonality": 12,
            "family": "gaussian",
            "maxiter": 200
        },
        model_transform_dict={
            "fillna": "nearest",
            "transformations": {
                "0": "StandardScaler"
            },
            'transformation_params': {
                "0": {}
            }
        },
        model_seasonality_dict={
            "seasonality": 12,
            "family": "gaussian",
            "maxiter": 200
        },
        df_train=df,
        forecast_length=7,
        frequency="D",
        prediction_interval=0.9,
        future_regressor_train = regr_train,
        future_regressor_forecast = regr_fcst,
        #no_negatives=False,
        # future_regressor_train=future_regressor_train2d,
        # future_regressor_forecast=future_regressor_forecast2d,
        #random_seed=321,
        verbose=0,
        n_jobs="auto",
    )
    print(df_forecast.forecast["revenue"].head(7))
    #3###############################################################################################
    """
    delta=0
    predict=7

    regr_train, regr_fcst = create_regressor(
        df, #df[:-7]
        forecast_length=predict,
        frequency=("D"),
        drop_most_recent=delta,
        scale=True,
        summarize="auto",
    )

    metric_weighting = { # В данном коде создается словарь metric_weighting, который содержит различные веса для различных метрик оценки модели.

                        #Каждая метрика оценки модели имеет свой вес, который отражает ее относительную важность в общей оценке модели. Веса метрик могут быть настроены в соответствии с требованиями и приоритетами задачи моделирования.

    #В данном случае, используется следующая система весов:

        'smape_weighting': 3, #3Вес для метрики SMAPE (симметричное среднее абсолютное процентное отклонение).
        'mae_weighting': 0, #5Вес для метрики MAE (среднее абсолютное отклонение).
        'rmse_weighting': 0, #1Вес для метрики RMSE (корень из среднеквадратичной ошибки).
        'made_weighting': 0, #1Вес для метрики MADE (среднее абсолютное процентное отклонение).
        'mage_weighting': 0, #0Вес для метрики MAGE (среднее абсолютное геометрическое отклонение).
        'mle_weighting': 0, #0Вес для метрики MLE (максимальная абсолютная ошибка).
        'imle_weighting': 0, #0Вес для метрики IMLE (инвертированная максимальная абсолютная ошибка).
        'spl_weighting': 0, #3Вес для метрики SPL (симметричное среднее процентное отклонение с логарифмической трансформацией).
        'dwae_weighting': 0, #1Вес для метрики DWAЕ (динамическое взвешенное абсолютное отклонение).
        'runtime_weighting': 0.05, #Вес для метрики времени выполнения модели.
    }
    initial_training = "auto"
    evolve = True
    """
    """
    gens=5 #5
    #Этот код определяет переменные, связанные с генетическим алгоритмом и настройками прогнозирования. Рассмотрим каждую строку по отдельности:
    if initial_training: #Если initial_training равно True, это означает, что файл шаблона не существует и происходит начальное обучение модели.
        gens = gens  #Эта переменная определяет количество поколений (generations) в генетическом алгоритме.
        generation_timeout = 10000  # minutes Эта переменная определяет время ограничения выполнения каждого поколения в генетическом алгоритме.
        models_to_validate = 0.15   #Эта переменная определяет долю моделей, которые будут оцениваться и выбираться для следующего поколения.
        ensemble = ["bestn"]  # , "mosaic", "mosaic-window", 'mlensemble' #Этот список определяет типы ансамблей моделей, которые будут использоваться в генетическом алгоритме
    elif evolve: #если значение переменной evolve равно True, и предыдущее условие initial_training равно False, то выполняется следующий блок кода. Если evolve равно True, это означает, что развитие временного ряда будет прогрессивным.
        gens = gens
        generation_timeout = 480  # minutes
        models_to_validate = 0.15
        ensemble = ["bestn"]  # "mosaic", "mosaic-window", "subsample"
    else: #Если ни одно из предыдущих условий не выполняется, то выполняется следующий блок кода. В этом случае развитие временного ряда не происходит, и используется фиксированный шаблон модели.
        gens = gens #Поскольку развитие временного ряда не происходит, нет необходимости выполнять генетический алгоритм, поэтому количество поколений (gens) устанавливается в 0.
        generation_timeout = 60  # minutes Поскольку генетический алгоритм не выполняется, время ограничения выполнения поколений устанавливается в 60 минут.
        models_to_validate = 0.99 #Поскольку генетический алгоритм не выполняется, все модели будут оценены и выбраны для использования.
        ensemble = ["bestn"] # "mosaic", "mosaic-window",


    initial_template = 'random'  # 'random' 'general+random'
    #Это определяет тип начального шаблона модели.

    preclean = None #Эта переменная определяет варианты предварительной обработки данных
    {  # preclean option
        "fillna": 'ffill', #десь определяется словарь параметров для предварительной очистки данных. В данном случае заданы параметры fillna и transformations.
        "transformations": {"0": "EWMAFilter"},
        "transformation_params": {
            "0": {"span": 14},
        },
    }

    forecast_name = "Ozone"
    n_jobs = "auto"

    model = AutoTS( #Создание объекта модели AutoTS.
        forecast_length=predict, #Установка длины прогноза (количество будущих периодов, на которые будет делаться прогноз).
        frequency=("D"), #Установка частоты данных (например, "D" для ежедневных данных).
        prediction_interval=(0.9), #Установка интервала прогноза, который определяет доверительный интервал для прогноза (например, [0.1, 0.9] для 80% доверительного интервала).
        #ensemble=["bestn"],
        ensemble=["bestn"],
        #ensemble=["horizontal-max", "dist", "simple"] , #Установка метода ансамблирования моделей.
        #model_list="fast_parallel_no_arima", #Установка списка моделей, которые будут использоваться в моделировании.
        model_list = "multivariate",
        transformer_list="fast", #Установка списка трансформеров (преобразований) данных, которые будут применяться к входным данным.
        transformer_max_depth=5, #Установка максимальной глубины применения трансформеров.
        max_generations=gens, #Установка максимального количества поколений (итераций) моделирования.
        metric_weighting=metric_weighting, #Установка весов для метрик оценки модели.
        initial_template=initial_template, #Установка начального шаблона модели (например, случайный или общий + случайный).
        aggfunc="sum", # Установка функции агрегации для группировки данных.
        models_to_validate=models_to_validate, #Установка доли моделей, которые будут оцениваться на каждом поколении.
        model_interrupt=True, #Установка флага, указывающего, разрешено ли прерывание моделирования.
        num_validations=(3), #Установка количества валидаций для каждой модели.
        validation_method="backwards", #Установка метода валидации моделей (например, "backwards" или "expanding").
        constraint=None, #Установка ограничений для моделей.
        drop_most_recent=delta,  # Установка флага, указывающего, нужно ли исключить самые последние данные (например, если они не полные).
        preclean=preclean, #Установка настроек предварительной обработки данных.
        models_mode="regressor", #Установка режима работы моделей (например, "default", "deep" или "regressor").
        # no_negatives=True,# no_negatives=True: Эта опция указывает, что модель не должна предсказывать отрицательные значения. Если у вас есть знания о вашем временном ряде, которые подразумевают отсутствие отрицательных значений, то можно использовать эту опцию для ограничения предсказаний модели только положительными значениями.
        # subset=100, #subset=100: Эта опция позволяет работать с подмножеством данных. Значение 100 указывает, что модель будет использовать только первые 100 точек данных из вашего временного ряда для моделирования и прогнозирования. Вы можете изменить это значение в соответствии с вашими потребностями.
        # prefill_na=0, #Эта опция определяет значение, которым будут заполняться пропущенные значения в данных перед моделированием. В данном случае, значение 0 указывает, что пропущенные значения будут заменены на ноль. Это может быть полезно, если вы хотите предоставить начальное значение для пропущенных точек данных перед применением модели.
        # remove_leading_zeroes=True, #Эта опция указывает, что модель должна удалить ведущие (начальные) нулевые значения в данных перед моделированием. Временные ряды иногда содержат начальные нулевые значения, которые могут быть неинформативными или не характерными для последующих точек данных. Установка этой опции в значение True позволяет удалить эти начальные нулевые значения для улучшения моделирования.
        current_model_file=f"current_model_{forecast_name}", #Установка имени файла для сохранения текущей модели.
        generation_timeout=generation_timeout, # Установка временного ограничения на выполнение поколения моделирования.
        n_jobs=n_jobs, #Установка количества параллельных задач для выполнения моделирования.
        verbose=1, #Установка уровня вывода информации о процессе моделирования (например, отображение прогресса).

    )



    #model = model.import_template("autots_forecast_template_Ozone_202307150039.csv", method='add_on', enforce_model_list = True, include_ensemble = True)
    #model = model.import_template("autots_forecast_template_Ozone_202307290554.csv", method='only') #лучшая модель с оркестрами
    #model = model.import_template("autots_forecast_template_Ozone_202307290554.csv", method='only')
    model = model.fit(
        df,
        future_regressor=regr_train, date_col= "time", value_col='revenue'
    )

    prediction = model.predict(
        future_regressor=regr_fcst, verbose=2, fail_on_forecast_nan=True
    )

    print(model)


    forecast_csv_name = None
    template_filename = f"autots_forecast_template_{forecast_name}.csv"
    if evolve:
        n_export = 1 #Это условное выражение проверяет, если значение переменной evolve равно True, то переменной n_export присваивается значение 1. Это означает, что будет сохраняться только самая лучшая модель во время прогрессивного развития временного ряда.
    else:
        n_export = 1  # wouldn't be a bad idea to do > 1, allowing some future adaptability Это означает, что будет сохраняться только самая лучшая модель, но без прогрессивного развития временного ряда.
    archive_templates = True  # save a copy of the model template used with a timestamp
    #Если установлено значение True, то будет сохранена копия используемого шаблона модели с отметкой времени



    forecasts_df = prediction.forecast  # .fillna(0).round(0) #Получение прогнозируемых значений из объекта prediction. forecasts_df будет содержать прогнозы модели.
    if forecast_csv_name is not None: #Проверка, указано ли имя файла для сохранения прогнозов в формате CSV.
        forecasts_df.to_csv(forecast_csv_name) #Если имя файла для сохранения прогнозов указано, то прогнозы сохраняются в указанный CSV-файл.

    forecasts_upper_df = prediction.upper_forecast #Получение верхней границы доверительного интервала прогнозов.
    forecasts_lower_df = prediction.lower_forecast #Получение нижней границы доверительного интервала прогнозов.

    # accuracy of all tried model results
    model_results = model.results() #Получение результатов моделирования, включая оценку точности всех протестированных моделей.
    validation_results = model.results("validation") #Получение результатов валидации моделей.

    # save a template of best models
    save_file = f"2_{template_filename.split('.csv')[0]}_{start_time.strftime('%Y%m%d%H%M')}.csv"
    if initial_training or evolve: #Проверка, было ли выполнено начальное обучение модели (initial_training) или разрешено ли постепенное развитие (evolve).
        model.export_template( #Сохранение шаблона лучших моделей в файл. Параметры n, max_per_model_class и include_results определяют количество сохраняемых моделей, максимальное количество моделей одного класса и включение результатов в файл шаблона.
            template_filename,
            models="best",
            n=n_export,
            max_per_model_class=6,
            include_results=True,
        )
        if archive_templates: #Проверка, разрешено ли архивирование шаблонов.
            arc_file = f"{template_filename.split('.csv')[0]}_{start_time.strftime('%Y%m%d%H%M')}.csv" #Формирование имени архивного файла шаблона с добавлением временной метки.
            model.export_template(arc_file, models="best", n=1) # Архивирование шаблона лучших моделей.
        model.export_template( #Сохранение шаблона лучших моделей в файл. Параметры n, max_per_model_class и include_results определяют количество сохраняемых моделей, максимальное количество моделей одного класса и включение результатов в файл шаблона.
            save_file,
            models="best",
            n=n_export,
            max_per_model_class=6,
            include_results=True,
        )


    print(f"Model failure rate is {model.failure_rate() * 100:.1f}%") #Вывод процента неудачных моделей.
    print(f'The following model types failed completely {model.list_failed_model_types()}') #Вывод типов моделей, которые полностью не справились с задачей.
    print("Slowest models:") #Вывод заголовка "Slowest models:".
    print(
        model_results[model_results["Ensemble"] < 1] #Вывод информации о самой медленной модели.
        .groupby("Model")
        .agg({"TotalRuntimeSeconds": ["mean", "max"]})
        .idxmax()
    )

    model_parameters = json.loads(model.best_model["ModelParameters"].iloc[0]) # Извлечение параметров лучшей модели.
     # model.export_template("all_results.csv", models='all')
    graph = False
    if graph: #Проверка, разрешена ли графическая визуализация.
        with plt.style.context("bmh"): #Контекстный менеджер для временного применения стиля "bmh" (blue-gray background with white gridlines) из библиотеки matplotlib.pyplot для последующих графиков, созданных внутри блока.
            start_date = 'auto'  # '2021-01-01' #Определение переменной start_date, которая указывает на начальную дату для графиков. Здесь установлено значение 'auto', что, вероятно, означает автоматическое определение начальной даты.
            #start_date = '2023-06-05'

            prediction.plot_grid(model.df_wide_numeric, start_date=start_date) #Создание сетки графиков прогнозов с использованием метода plot_grid объекта prediction. Входные данные для графика берутся из model.df_wide_numeric, а начальная дата задается параметром start_date.
            plt.show() #Отображение созданного графика.

            worst = model.best_model_per_series_score().head(6).index.tolist() #Получение списка индексов (названий) наихудших моделей с помощью метода best_model_per_series_score() объекта model. В данном случае, выбираются первые 6 наихудших моделей.
            prediction.plot_grid(model.df_wide_numeric, start_date=start_date, title="Worst Performing Forecasts", cols=worst) #Создание сетки графиков прогнозов для наихудших моделей. Параметр title задает заголовок графика, а cols указывает столбцы данных, соответствующие наихудшим моделям.
            plt.show() #Отображение созданного графика.

            best = model.best_model_per_series_score().tail(6).index.tolist() #Получение списка индексов (названий) наилучших моделей с помощью метода best_model_per_series_score() объекта model. В данном случае, выбираются последние 6 наилучших моделей.
            prediction.plot_grid(model.df_wide_numeric, start_date=start_date, title="Best Performing Forecasts", cols=best) # Создание сетки графиков прогнозов для наилучших моделей. Параметр title задает заголовок графика, а cols указывает столбцы данных, соответствующие наилучшим моделям.
            plt.show() #Отображение созданного графика.

            if model.best_model_name == "Cassandra": #Это условие проверяет, является ли лучшая модель названной "Cassandra". Если это так, то выполняется следующий блок кода, который относится к визуализации компонентов модели и тренда.
                prediction.model.plot_components( #Создание графиков компонентов модели с помощью метода plot_components объекта prediction.model. Параметр series указывает на ряды данных, to_origin_space указывает, следует ли преобразовать значения в исходное пространство (например, возвращение к исходным масштабам), а start_date задает начальную дату.
                    prediction, series=None, to_origin_space=True, start_date=start_date
                )
                plt.show()
                prediction.model.plot_trend( #Создание графика тренда с помощью метода plot_trend объекта prediction.model. Параметр series указывает на ряды данных, а start_date задает начальную дату.
                    series=None, start_date=start_date
                )
                plt.show()

            ax = model.plot_per_series_mape() #Создание графика средней абсолютной процентной ошибки для каждого ряда данных с использованием метода plot_per_series_mape объекта model. Результат сохраняется в переменную ax.
            plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0) #Добавление легенды к графику, указывающей на значения рядов данных. bbox_to_anchor, loc и borderaxespad устанавливают позицию и отступы легенды.
            plt.show()

            back_forecast = False
            if back_forecast: #Это условие проверяет, разрешено ли обратное прогнозирование. Если это так, выполняется следующий блок кода, который относится к визуализации обратного прогнозирования.
                model.plot_backforecast() #Создание графика обратного прогнозирования с помощью метода plot_backforecast объекта model.
                plt.show()

            ax = model.plot_validations()
            plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0) #Добавление легенды к графику, указывающей на значения рядов данных. bbox_to_anchor, loc и borderaxespad устанавливают позицию и отступы легенды.
            plt.show()

            ax = model.plot_validations(subset='best') #Создание графика валидации для лучших моделей с помощью метода plot_validations объекта model. Параметр subset устанавливает, какие модели включить в график.
            plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0) #Добавление легенды к графику.
            plt.show()

            ax = model.plot_validations(subset='worst') #Создание графика валидации для наихудших моделей с помощью метода plot_validations объекта model. Параметр subset устанавливает, какие модели включить в график.
            plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0) #Добавление легенды к графику.
            plt.show()

            if model.best_model_ensemble == 2: #Проверка, является ли лучшая модель ансамблем с двумя моделями. Если условие выполняется, выполняется следующий блок кода.
                plt.subplots_adjust(bottom=0.5) #Изменение расположения подграфиков, чтобы снизу было больше места.
                model.plot_horizontal_transformers() #Создание графика, показывающего количество горизонтальных трансформеров.
                plt.show()
                model.plot_horizontal_model_count() #Создание графика, показывающего количество моделей в ансамбле.
                plt.show()

                model.plot_horizontal() #Создание горизонтального графика, который показывает сравнение производительности различных моделей.
                plt.show()
                # plt.savefig("horizontal.png", dpi=300, bbox_inches="tight")

                if str(model_parameters["model_name"]).lower() in ["mosaic", "mosaic-window"]: #Проверка, является ли лучшая модель мозаикой или мозаикой с окном. Если это так, выполняется следующий блок кода.
                    mosaic_df = model.mosaic_to_df() #Преобразование мозаики в DataFrame с помощью метода mosaic_to_df объекта model.
                    print(mosaic_df[mosaic_df.columns[0:5]].head(5)) #Вывод первых пяти строк DataFrame с мозаикой, ограниченных первыми пятью столбцами.

    print(f"Completed at system time: {datetime.datetime.now()}")

    df_forecast = prediction.forecast

    print(df_forecast["revenue"].head(predict))

    n_back=delta #сколько параметров с конца откусываем
    start=combined_df['revenue'].iloc[-1-n_back]
    print("Стартовое значения для восстановления=",start)

    df_pred = forecasts_df["revenue"].copy()
    df_pred=df_pred.to_frame()
    print(df_pred)
    df_pred.reset_index(inplace=True)
    df_pred.rename(columns={"index": "Data"}, inplace=True)
    df_pred['Data'] = pd.to_datetime(df_pred['Data'])
    df_pred['revenue'] = df_pred['revenue'].cumsum() + start


    #######################################################################
    ##########################  Определение ошибки #############################################
    try:
        with open(template_filename, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                smape = row['smape']
                rmse = row['rmse']
                print("Ошибка smape =", smape)
                break  # Чтобы прекратить чтение после первой строки
    except FileNotFoundError:
        print(f"Файл '{template_filename}' не найден.")
    except csv.Error as e:
        print(f"Ошибка при работе с файлом CSV: {e}")

    print(f"Имя файла '{save_file}")
    print(f"Имя файла2 '{arc_file}")

    import time
    from datetime import datetime

    file_name = r'C:\Users\Andrey\Desktop\python\Ozon_models_history.txt'
    current_timestamp = int(time.time())
    formatted_time = datetime.fromtimestamp(current_timestamp).strftime("%Y-%m-%d %H:%M:%S")

    with open(file_name, 'a') as file:
        file.write(
            f"{formatted_time},{save_file},{smape},{rmse}\n")

    ##############################################################################


    import matplotlib.pyplot as plt
    # Создаем копию датафрейма data
    data_copy2 = combined_df.copy()

    # Оставляем только столбцы "id" (переименованный в "Data") и "Заказано на сумму"
    #data_copy = data_copy[["id", "revenue"]]

    # Переименовываем столбец "id" в "Data"
    data_copy2.rename(columns={"time": "Data"}, inplace=True)
    print (data_copy2)
    print (df_pred)


    data_copy2['Data'] = pd.to_datetime(data_copy2['Data'])

    # Отображение последних 14 строк в data_copy
    data_copy_last_14 = data_copy2.tail(40)

    # Создание нового графического объекта и установка размеров
    plt.figure(figsize=(10, 5))  # Указывает размеры в дюймах

    # Построение графика
    plt.plot(df_pred['Data'], df_pred['revenue'], label='Predict')
    plt.plot(data_copy_last_14['Data'], data_copy_last_14['revenue'], label='Actual')

    # Настройка осей и легенды
    plt.xlabel('Data')
    plt.ylabel('Revenue')
    plt.legend()

    # Отображение графика
    plt.show()

    """
    
    # Создание модели AutoTS
    model_High = AutoTS(
        forecast_length=2,
        frequency='infer',
        ensemble='bestn',
        #model_list="fast",  # "superfast", "default", "fast_parallel"
        model_list=list_of_models,
        transformer_list="fast",  # "superfast",
        drop_most_recent=0, #число последних точек, которые следует удалить из данных перед обучением модели. Здесь установлено значение 1.
        max_generations=4,
        num_validations=2,
        validation_method="backwards")
    
    # Оставляем только столбец 'Low' в DataFrame
    df_High = df['EURUSD_High']
    
    # Обучение модели
    model_High = model_High.fit(df_diff, date_col= "Opentime", value_col='EURUSD_High')
    
    
    # Создание модели AutoTS
    model_Low = AutoTS(
        forecast_length=2,
        frequency='infer',
        ensemble='bestn',
        #model_list="fast",  # "superfast", "default", "fast_parallel"
        model_list=list_of_models,
        transformer_list="fast",  # "superfast",
        drop_most_recent=0, #число последних точек, которые следует удалить из данных перед обучением модели. Здесь установлено значение 1.
        max_generations=4,
        num_validations=2,
        validation_method="backwards")
    
    # Оставляем только столбец 'Low' в DataFrame
    df_low = df['EURUSD_Low']
    
    # Обучение модели
    model_Low = model_Low.fit(df_diff, date_col= "Opentime", value_col='EURUSD_Low')
    
    print("model_High,model_Low=",model_High,model_Low)
    
    # Предсказание
    prediction_High = model_High.predict()
    prediction_Low = model_Low.predict()
    # Прогнозные данные
    forecast_High = prediction_High.forecast
    forecast_Low = prediction_Low.forecast
    
    print("forecast_High=",forecast_High)
    print("forecast_Low=",forecast_Low)
    
    
    
    
    # upper and lower forecasts
    forecasts_High_up, forecasts_High_low = prediction_High.upper_forecast, prediction_High.lower_forecast
    print("Максимальное значение предикта=",forecasts_High_up)
    print("Миниимальное значение предикта=",forecasts_High_low)
    
    
    # accuracy of all tried model results
    model_High.results()
    
    # and aggregated from cross validation
    model_High.results("model_weights")
    
    
    # Обратное дифференцирование значений
    restored_forecasts = forecast_High.cumsum() + combined_df['EURUSD_High'].iloc[-1]
    
    # Вывод восстановленных значений
    print("Вывод восстановленных значений forecast_High=",restored_forecasts)
    
    # Обратное дифференцирование значений
    restored_forecasts_low = forecast_Low.cumsum() + combined_df['EURUSD_Low'].iloc[-1]
    
    # Вывод восстановленных значений
    print("Вывод восстановленных значений forecast_Low=",restored_forecasts_low)
    
    name_h = model_High.best_model_name
    name_l = model_Low.best_model_name
    
    model_High.best_model_params
    
    model_High.best_model_transformation_params
    
    gap = 0.00030
    
    print("forecast_High['EURUSD_High']=",forecast_High['EURUSD_High'])
    print("forecast_Low['EURUSD_Low']=",forecast_Low['EURUSD_Low'])
    
    value = np.add(forecast_High['EURUSD_High'], forecast_Low['EURUSD_Low'])
    trend_direction = np.where(value > gap, 2, np.where(value < -gap, 1, 0))
    print("value=",value)
    
    def get_trend_direction(forecast_High, forecast_Low):
        value = np.add(forecast_High['EURUSD_High'], forecast_Low['EURUSD_Low'])
        trend_direction = np.where(value > gap, 2, np.where(value < -gap, 1, 0))
        return trend_direction
    
    
    def get_value(trend_direction):
        if np.all(trend_direction == 2):
            return 2
        elif np.any(trend_direction == 1):
            return 1
        elif np.any(trend_direction == 0):
            return 0
        else:
            return 0
    
    get_trend = get_trend_direction(forecast_High, forecast_Low)
    get_trend
    
    get_value(get_trend)
    
    def get_trend_direction(forecast_High, forecast_Low):
        trend_direction_High = np.where(forecast_High['EURUSD_High'] > gap, 2, np.where(forecast_High['EURUSD_High'] < -gap, 1, 0))
        trend_direction_Low = np.where(forecast_Low['EURUSD_Low'] > gap, 2, np.where(forecast_Low['EURUSD_Low'] < -gap, 1, 0))
        combined_trend_direction = np.concatenate((trend_direction_High, trend_direction_Low))
        print(combined_trend_direction)
        return combined_trend_direction
    
    trend_direction_High = get_trend_direction(forecast_High, forecast_Low)
    
    print(name_h)
    print(name_l)
    print(trend_direction_High)
    
    if all(x == 2 for x in trend_direction_High):
        signal = 2
    elif all(x == 1 for x in trend_direction_High):
        signal = 1
    else:
        signal = 0
    
    print(signal)
    """
    #file_path = r'C:\Users\Andrey\AppData\Roaming\MetaQuotes\Terminal\FA97EA291D4188820508F9D2B5AAD50F\MQL5\Files\file_signal.txt'

    """
    with open(file_path, 'w') as file:
        #file.write(str(signal))
        file.write(f"{name_h},{name_l},{trend_direction_High},{signal}\n")
    """

    """
    with open(file_path, 'a') as file:
        file.write(f"{name_h},{name_l},{trend_direction_High},{signal},1\n")
    
    
    
    file_path = os.path.join(drive_path, 'Models.txt')
    # Проверяем, существует ли файл
    if os.path.exists(file_path):
        # Открываем файл в режиме добавления данных в конец
        with open(file_path, 'a') as file:
            # Записываем переменные на новые строки
            file.write(f'{name_h}\n{name_l}\n')
    else:
        # Создаем новый файл и записываем переменные
        with open(file_path, 'w') as file:
            file.write(f'{name_h}\n{name_l}\n')
    """


    """
    
    # plot a sample
    prediction_High.plot(model_High.df_wide_numeric,
                    series=model_High.df_wide_numeric.columns[0],
                    start_date="2023-07-04",
                    figsize=(7, 4))
    
    
    # plot a sample
    prediction_Low.plot(model_Low.df_wide_numeric,
                    series=model_Low.df_wide_numeric.columns[0],
                    start_date="2023-07-04",
                    figsize=(7, 4))
    plt.show()
    """
    """
    Для запуска задачи каждый час и 1 минуту можно воспользоваться типом запуска cron в APScheduler. В cron-формате вы можете указать расписание с помощью выражений времени.
    
    Вот пример кода, который позволит вам выполнять задачу каждый час и 1 минуту:
    
    python
    from apscheduler.schedulers.blocking import BlockingScheduler
    
    def job_function():
        print("Hello, World!")
    
    # Создаем экземпляр планировщика
    scheduler = BlockingScheduler()
    
    # Добавляем задачу, которую нужно выполнить каждый час и 1 минуту
    scheduler.add_job(job_function, 'cron', hour='*', minute='1')
    
    # Запускаем планировщик
    scheduler.start()
    В данном примере мы использовали выражение 'cron' в качестве типа запуска. Параметры hour='*' и minute='1' указывают, что задача должна выполняться каждый час ('*' означает любое значение для часов) и в 1 минуту.
    
    Вы также можете настроить другие параметры времени, такие как дни недели (day_of_week=), дни месяца (day=) и месяцы (month=), чтобы создать более сложное расписание, если это необходимо.
    
    Примечание: Запуск каждую минуту после полного часа (например, в 12:01, 13:01 и т. д.) может быть более надежным, чем запуск в ровные 1 минуту. Это связано с возможной задержкой запуска и точностью системных часов.
    """

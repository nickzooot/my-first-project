|test| |codecov| |docs|

.. |test| image:: https://github.com/intsystems/ProjectTemplate/workflows/test/badge.svg
    :target: https://github.com/intsystems/ProjectTemplate/tree/master
    :alt: Test status
    
.. |codecov| image:: https://img.shields.io/codecov/c/github/intsystems/ProjectTemplate/master
    :target: https://app.codecov.io/gh/intsystems/ProjectTemplate
    :alt: Test coverage
    
.. |docs| image:: https://github.com/intsystems/ProjectTemplate/workflows/docs/badge.svg
    :target: https://intsystems.github.io/ProjectTemplate/
    :alt: Docs status


.. class:: center

    :Название исследуемой задачи: Детекция эмоций. Сравнение и анализ классических методов машинного обучения и методов обучения с трансформерами.
    :Тип научной работы: M1P
    :Автор: Панин Никита Александрович 
    :Научный руководитель: д.ф-м.н, профессор, Воронцов Константин Вячеславович
    :Научный консультант(при наличии): -

Abstract
========

В работе была рассмотрена задача детекции эмоций на датасете, в ос-
нову которого вошел WASSA датасет из твитов для детекции эмоций.
На выходе алгоритма классификации эмоций в твитах была одна из 5
эмоций: нейтральная эмоция, грусть, страх, радость, гнев. Были при-
менены различные методы "классического" машинного обучения, такие
как, SVM, логистическая регрессия, метод k-ближайших соседей и на-
ивный байесовский классификатор. Также классификация эмоций была
проведена с помощью файн-тюнинга нескольких версий BERT. Основной
целью работы являлось проведение сравнительного анализа для класси-
ческих моделей машинного обучения(wKNN, Multinomial Bayes Classifier,
Logistic Regression, SVM) и для моделей глубокого обучения (в качестве
предобученной модели брались BERT, RoBERTa, BERTweet и их large-
версии)
В результате исследования было показано, что по метрике accuracy
для моделей классического обучения c tf-idf векторизацией текстов луч-
ше всего работает SVM с RBF ядром (accuracy ≈ 0.8387 на тесте), а
наиболее качественные результаты получаются с помощью предложен-
ной в исследовании модели с предобученным BERTweet (accuracy ≈0.88
на тесте)м.

Research publications
===============================
1. 

Presentations at conferences on the topic of research
================================================
1. 

Software modules developed as part of the study
======================================================
1. A python package *mylib* with all implementation `here <https://github.com/intsystems/ProjectTemplate/tree/master/src>`_.
2. A code with all experiment visualisation `here <https://github.comintsystems/ProjectTemplate/blob/master/code/main.ipynb>`_. Can use `colab <http://colab.research.google.com/github/intsystems/ProjectTemplate/blob/master/code/main.ipynb>`_.

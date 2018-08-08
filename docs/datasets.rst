.. _datasets:

Datasets
========

A few example datasets are distributed within WrightTools.
These make it easy to demonstrate and test data processing features.
They're also a lot of fun!

The following table contains every dataset distributed within WrightTools.

=================================================  ============================  ===================  ==============
dataset                                            axis expressions              shape                gallery links
=================================================  ============================  ===================  ==============
``BrunoldrRaman.LDS821_514nm_80mW``                ``('energy',)``               ``(1340,)``          :ref:`sphx_glr_auto_examples_rRaman.py` 
``Cary.CuPCtS_H2O_vis (collection)``               ``('wavelength',)``           ``(141,)``
``Cary.filters (collection)``                      ``('wavelength',)``           ``multiple``         :ref:`sphx_glr_auto_examples_filters.py`
``COLORS.v0p2_d1_d2_diagonal`` [#kohler2014]_      ``('d1', 'd2')``              ``(21, 21)``         :ref:`sphx_glr_auto_examples_fill_types.py`
``COLORS.v2p2_WL_wigner``                          ``('wm', 'd1')``              ``(241, 51)``        
``JASCO.PbSe_batch_1`` [#yurs2011]_                ``('energy',)``               ``(1801,)``      
``JASCO.PbSe_batch_4_2012_02_21`` [#kohler2014]_   ``('energy',)``               ``(1251,)``     
``JASCO.PbSe_batch_4_2012_03_15`` [#kohler2014]_   ``('energy',)``               ``(1251,)``    
``KENT.LDS821_DOVE`` [#neffmallon2017]_            ``('w2', 'w1')``              ``(60, 60)``         :ref:`sphx_glr_auto_examples_DOVE_transform.py`
``KENT.LDS821_TRSF`` [#boyle2013]_                 ``('w2', 'w1')``              ``(71, 71)``         :ref:`sphx_glr_auto_examples_quick2D.py`, :ref:`sphx_glr_auto_examples_quick1D.py` 
``KENT.PbSe_2D_delay_B`` [#yurs2011]_              ``('d2', 'd1')``              ``(101, 101)``
``ocean_optics.tsunami``                           ``('energy',)``               ``(2048,)``
``PyCMDS.d1_d2_000``                               ``('d1', 'd2')``              ``(101, 101)``       :ref:`sphx_glr_auto_examples_label_delay_space.py`
``PyCMDS.d1_d2_001``                               ``('d1', 'd2')``              ``(101, 101)``       :ref:`sphx_glr_auto_examples_label_delay_space.py`
``PyCMDS.w1_000``                                  ``('w1',)``                   ``(51,)``
``PyCMDS.w1_wa_000``                               ``('w1=wm', 'wa')``           ``(25, 256)``        :ref:`sphx_glr_auto_examples_tune_test.py`
``PyCMDS.w2_w1_000`` [#morrow2017]_                ``('w2', 'w1')``              ``(81, 81)``         :ref:`sphx_glr_auto_examples_fringes_transform.py`
``PyCMDS.wm_w2_w1_000``                            ``('wm', 'w2', 'w1')``        ``(35, 11, 11)``
``PyCMDS.wm_w2_w1_001``                            ``('wm', 'w2', 'w1')``        ``(29, 11, 11)``
``Shimadzu.MoS2_fromCzech2015`` [#czech2015]_      ``('energy',)``               ``(819,)``
``Solis.wm_ypos_fluorescence_with_filter``         ``('wm', 'ypos')``            ``(2560, 2160)``
``Solis.xpos_ypos_fluorescence``                   ``('xpos', 'ypos')``          ``(2560, 2160)``
``spcm.test_data``                                 ``('time',)``                 ``(1024,)``
``spcm.test_data_full_metadata``				   ``('time',)``				 ``(1024,)``
``Tensor27.CuPCtS_powder_ATR``                     ``('energy',)``               ``(7259,)``
``wt5.v1p0p0_perovskite_TA``                       ``('w1=wm', 'w2', 'd2')``     ``(52, 52, 13)``     :ref:`sphx_glr_auto_examples_quick2d_signed.py`
``wt5.v1p0p1_MoS2_TrEE_movie`` [#czech2015]_       ``('w2', 'w1', 'd2')``        ``(41, 41, 23)``     :ref:`sphx_glr_auto_examples_level.py`, :ref:`sphx_glr_auto_examples_colormaps.py`
=================================================  ============================  ===================  ==============

.. [#boyle2013] **Triply Resonant Sum Frequency Spectroscopy: Combining Advantages of Resonance Raman and 2D-IR**
                Erin S. Boyle, Nathan A. Neff-Mallon, and John C. Wright
                *The Journal of Physical Chemistry A* **2013** 117 (47), 12401-12408
                `doi:10.1021/jp409377a <http://dx.doi.org/10.1021/jp409377a>`_

.. [#czech2015] **Measurement of Ultrafast Excitonic Dynamics of Few-Layer MoS2 Using State-Selective Coherent Multidimensional Spectroscopy**
                Kyle J. Czech, Blaise J. Thompson, Schuyler Kain, Qi Ding, Melinda J. Shearer, Robert J. Hamers, Song Jin, and John C. Wright
                *ACS Nano* **2015** 9 (12), 12146-12157
                `doi:10.1021/acsnano.5b05198 <http://dx.doi.org/10.1021/acsnano.5b05198>`_

.. [#kohler2014] **Ultrafast Dynamics within the 1S Exciton Band of Colloidal PbSe Quantum Dots Using Multiresonant Coherent Multidimensional Spectroscopy**
                 Daniel D. Kohler, Stephen B. Block, Schuyler Kain, Andrei V. Pakoulev, and John C. Wright
                 *The Journal of Physical Chemistry C* **2014** 118 (9), 5020-5031
                 `doi:10.1021/jp412058u <http://dx.doi.org/10.1021/jp412058u>`_

.. [#morrow2017] **Group and phase velocity mismatch fringes in triple sum-frequency spectroscopy**
                 Darien J. Morrow, Daniel D. Kohler, and John C. Wright
                 *Physical Review A* **2017** 96, 063835
                 `doi:10.1103/PhysRevA.96.063835 <http://dx.doi.org/10.1103/PhysRevA.96.063835>`_

.. [#neffmallon2017] **Multidimensional Spectral Fingerprints of a New Family of Coherent Analytical Spectroscopies**
                 Nathan A. Neff-Mallon and John C. Wright
                 *Analytical Chemistry* **2017** 89 (24), 13182–13189
                 `doi:10.1021/acs.analchem.7b02917 <http://dx.doi.org/10.1021/acs.analchem.7b02917>`_

.. [#yurs2011] **Multiresonant Coherent Multidimensional Electronic Spectroscopy of Colloidal PbSe Quantum Dots**
               Lena A. Yurs, Stephen B. Block, Andrei V. Pakoulev, Rachel S. Selinsky, Song Jin, and John Wright
               *The Journal of Physical Chemistry C* **2011** 115 (46), 22833-22844
               `doi:10.1021/jp207273x <http://dx.doi.org/10.1021/jp207273x>`_


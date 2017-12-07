.. _datasets:

Datasets
========

A few example datasets are distributed within WrightTools.
These make it easy to demonstrate and test data processing features.
They're also a lot of fun!

The following table contains every dataset distributed within WrightTools.

=================================================  ============================  ===================  ==============
dataset                                            axis names                    shape                gallery links
=================================================  ============================  ===================  ==============
``BrunoldrRaman.LDS821_514nm_80mW``                ``['wm']``                    ``(1340,)``
``Cary50.CuPCtS_H2O_vis``                          ``['wm']``                    ``(141,)``
``COLORS.v0p2_d1_d2_diagonal`` [#kohler2014]_      ``['d1', 'd2']``              ``(21, 21)``        
``COLORS.v0p2_d1_d2_off_diagonal`` [#kohler2014]_  ``['d1', 'd2']``              ``(21, 21)``       
``COLORS.v2p1_MoS2_TrEE_movie`` [#czech2015]_      ``['w2', 'w1', 'd2']``        ``(41, 41, 23)``  
``JASCO.PbSe_batch_1`` [#yurs2011]_                ``['wm']``                    ``(1801,)``      
``JASCO.PbSe_batch_4_2012_02_21`` [#kohler2014]_   ``['wm']``                    ``(1251,)``     
``JASCO.PbSe_batch_4_2012_03_15`` [#kohler2014]_   ``['wm']``                    ``(1251,)``    
``KENT.LDS821_TRSF`` [#boyle2013]_                 ``['w2', 'w1']``              ``(71, 71)``         
``KENT.PbSe_2D_delay_A`` [#yurs2011]_              ``['d2', 'd1']``              ``(101, 151)``         
``KENT.PbSe_2D_delay_B`` [#yurs2011]_              ``['d2', 'd1']``              ``(101, 101)``         
``PyCMDS.w2_w1_000`` [#morrow2017]_                ``['w2', 'w1']``              ``(81, 81)``         :ref:`sphx_glr_auto_examples_fringes_transform.py`
``PyCMDS.wm_w2_w1_000``                            ``['wm', 'w2', 'w1']``        ``(35, 11, 11)``
``PyCMDS.wm_w2_w1_001``                            ``['wm', 'w2', 'w1']``        ``(29, 11, 11)``
``spcm.test_data``                                 ``['time']``                  ``(1024,)``
``Tensor27.CuPCtS_powder_ATR``                     ``['w']``                     ``(7259,)``
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
                 `arXiv:1709.10476 <https://arxiv.org/abs/1709.10476>`_

.. [#yurs2011] **Multiresonant Coherent Multidimensional Electronic Spectroscopy of Colloidal PbSe Quantum Dots**
               Lena A. Yurs, Stephen B. Block, Andrei V. Pakoulev, Rachel S. Selinsky, Song Jin, and John Wright
               *The Journal of Physical Chemistry C* **2011** 115 (46), 22833-22844
               `doi:10.1021/jp207273x <http://dx.doi.org/10.1021/jp207273x>`_


#ifndef WIRECELL_GENSTDPAR_IMPACTTRANSFORM
#define WIRECELL_GENSTDPAR_IMPACTTRANSFORM

#include "WireCellGenStdpar/BinnedDiffusion_transform.h"
#include "WireCellIface/IPlaneImpactResponse.h"
#include "WireCellIface/IDFT.h"
#include "WireCellUtil/Array.h"
//#include "WireCellUtil/Logging.h"

#include <Eigen/Sparse>

namespace WireCell {
    namespace GenStdpar {

        /** An ImpactTransform transforms charge on impact positions
         * into waveforms via 2D FFT.
         */
        class ImpactTransform {
            IPlaneImpactResponse::pointer m_pir;
            IDFT::pointer m_dft;
            BinnedDiffusion_transform& m_bd;

            int m_num_group;     // how many 2D convolution is needed
            int m_num_pad_wire;  // how many wires are needed to pad on each side
            std::vector<std::map<int, IImpactResponse::pointer> > m_vec_map_resp;
            std::vector<std::vector<std::tuple<int, int, double> > > m_vec_vec_charge;  // ch, time, charge
            // std::vector<Eigen::SparseMatrix<float>* > m_vec_spmatrix;

            std::vector<int> m_vec_impact;
            Array::array_xxf m_decon_data;
            int m_start_ch;
            int m_end_ch;
            int m_start_tick;
            int m_end_tick;

//            Log::logptr_t log;

           public:
            ImpactTransform(IPlaneImpactResponse::pointer pir, const IDFT::pointer& dft, BinnedDiffusion_transform& bd);
            virtual ~ImpactTransform();

            bool transform_vector();

            bool transform_matrix();

            /// Return the wire's waveform.  If the response functions
            /// are just field response (ie, instantaneous current)
            /// then the waveforms are expressed as current integrated
            /// over each sample bin and thus in units of charge.  If
            /// the response functions include electronics response
            /// then the waveforms are in units of voltage
            /// representing the sampling of the output of the FEE
            /// amplifiers.

            // fixme: this should be a forward iterator so that it may cal bd.erase() safely to conserve memory
            Waveform::realseq_t waveform(int wire) const;
        };

    }  // namespace GenStdpar
}  // namespace WireCell
#endif /* WIRECELL_GENSTDPAR_IMPACTTRANSFORM */

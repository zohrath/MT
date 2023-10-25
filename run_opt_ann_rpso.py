# Run the RPSO to optimize the weights and biases of an ANN
from __future__ import division
import sys
import time
import numpy as np
import multiprocessing
from functools import partial
import pandas as pd
import tensorflow as tf
from CostFunctions import get_fingerprinted_data, get_fingerprinted_data_noisy

from RPSO import RPSO
from Statistics import save_opt_ann_rpso_stats
from pso_options import create_model


X_train, _, y_train, _, _ = get_fingerprinted_data()
model, num_dimensions = create_model()


def ann_weights_fitness_function(particle, model):
    for layer in model.layers:
        weights = layer.get_weights()[0]
        biases = layer.get_weights()[1]
        num_weights = weights.size
        num_biases = biases.size

        # Slice off values from the continuous_values array for weights and biases
        sliced_weights = particle[:num_weights]
        sliced_biases = particle[num_weights: num_weights + num_biases]

        # Update the continuous_values array for the next iteration
        particle = particle[num_weights + num_biases:]

        # Set the sliced weights and biases in the layer
        layer.set_weights(
            [sliced_weights.reshape(weights.shape),
             sliced_biases.reshape(biases.shape)]
        )
    try:
        model.compile(optimizer="adam", loss="mse")

        # Evaluate the model and get the evaluation metrics
        evaluation_metrics = model.evaluate(X_train, y_train, verbose=1)
        rmse = np.sqrt(evaluation_metrics)

        return rmse
    except tf.errors.InvalidArgumentError as e:
        # Handle the specific exception here
        print("Caught an InvalidArgumentError:", e)
        # You can choose to return a specific value or take other actions
        return float("inf")  # For example, return infinity in case of an error
    except tf.errors.OpError as e:
        # Handle TensorFlow-specific errors here
        print(f"TensorFlow error: {e}")
        return float("inf")  # For example, return infinity in case of an error
    except Exception as e:
        # Handle other exceptions here
        print(f"An error occurred: {e}")
        return float("inf")  # For example, return infinity in case of an error


def run_pso(
    _,
    iterations,
    position_bounds,
    velocity_bounds,
    fitness_threshold,
    num_particles,
    Cp_min,
    Cp_max,
    Cg_min,
    Cg_max,
    w_min,
    w_max,
    gwn_std_dev,
):
    swarm = RPSO(
        iterations,
        num_particles,
        num_dimensions,
        position_bounds,
        velocity_bounds,
        Cp_min,
        Cp_max,
        Cg_min,
        Cg_max,
        w_min,
        w_max,
        fitness_threshold,
        ann_weights_fitness_function,
        gwn_std_dev,
    )

    swarm.run_pso(model)

    return (
        swarm.swarm_best_fitness,
        swarm.swarm_best_position,
        swarm.swarm_fitness_history,
        swarm.swarm_position_history,
    )


if __name__ == "__main__":
    # ---- RSPO options ----
    position_bounds = [(-4.0, 4.0)] * num_dimensions
    velocity_bounds = [(-0.2, 0.2)] * num_dimensions
    fitness_threshold = 0.1
    num_particles = 30
    # Cp_min = 0.1
    # Cp_max = 3.5
    # Cg_min = 0.9
    # Cg_max = 3.0
    # w_min = 0.3
    # w_max = 1.7
    # gwn_std_dev = 0.15
    iterations = 200

    velocity_bounds = [0.542026628219197, 0.6660427138575113, 0.7583292050727198, 0.22221780704215183, 0.35052810009770774, 0.4604767638547535, 0.1604997494619848, 0.08715369383232649, 0.30894726707559683, 0.0902293995771882, 0.678326566735701, 0.7540834856853711, 0.6068428849586925, 0.6845797265866946, 0.05892978363517824, 0.1070677312023772, 0.6625221909316715, 0.6105218144387835, 0.2020181679436405, 0.18426498298241933, 0.23129792320819204, 0.4682160800378128, 0.24200231994757446, 0.41892167513878364, 0.5285966709504937, 0.07101002338571276, 0.2985448352910976, 0.6344446868621303, 0.7577554425034048, 0.6500080112705058, 0.7419290351893216, 0.7895078312635314, 0.665012982470186, 0.36269964737349913, 0.2188337996069165, 0.5161766653136797, 0.08566955340408122, 0.4029473749399326, 0.18997085425932697, 0.45258061022571466, 0.05495680241565093, 0.11273124701081763, 0.08005359698442467, 0.41216654059748536, 0.1985752525495775, 0.6253845176208191, 0.24496789955620688, 0.2843012062838127, 0.35782761377437783,
                       0.6689741586105914, 0.1475915134553054, 0.5197450012212889, 0.18120062315165428, 0.6188364866684649, 0.7686896272208089, 0.6030050255606946, 0.0683587500257706, 0.28436150993353215, 0.5129845560285248, 0.04178873042906113, 0.2807111089737009, 0.7459093007137884, 0.2756454212648537, 0.06481016273420334, 0.1683583520928867, 0.6445915667171177, 0.621536418779269, 0.1847366255345066, 0.5224365598234585, 0.5727168518861737, 0.29007289662262914, 0.09637830567992264, 0.34149766685362226, 0.25075366125782467, 0.6164545330679831, 0.2610572949023807, 0.43649972014578836, 0.5081047551068012, 0.07424351354149411, 0.5006944819730034, 0.732732633737257, 0.041396298498333026, 0.1729377281516451, 0.5669769898875903, 0.49305341516053525, 0.1856638700653086, 0.6224193478649852, 0.274297540723357, 0.05956926693752715, 0.07952712854296815, 0.3771574773872123, 0.29493499477033314, 0.272718360996686, 0.6406467824612986, 0.7592455059645575, 0.6894299055029663, 0.4555071194724097, 0.6345490629278445, 0.5544357465223577, 0.7428206631023622]
    velocity_bounds = [[(-x, x)] * num_dimensions for x in velocity_bounds]

    Cp_min = [0.5278968313898457, 0.8527619538134347, 0.3764754221528622, 0.3431970463865234, 0.4887118903424523, 0.8203243391197244, 0.5285564395933434, 0.18438468973530292, 0.2659839387751669, 0.2592519554741033, 0.4587871452574681, 0.6343531529450198, 0.4111465192499625, 0.5051584165487223, 0.9296847465997541, 0.2223007930381687, 0.9386981859907751, 0.48842614668454376, 0.6697208808415132, 0.26168619416827804, 0.6722461962726685, 0.2851074613953716, 0.6221587085170952, 0.6943281091051517, 0.6323712812613951, 0.7852101859584124, 0.8049319920747168, 0.8660809954249515, 0.749528870969856, 0.43032520406021935, 0.9200274336068488, 0.21055296122158168, 0.2965337319629344, 0.6514013795884048, 0.4621825989359495, 0.9159047882912491, 0.32512805767677977, 0.7538785229928376, 0.42285475683515117, 0.39454556861109147, 0.7276233766790366, 0.5002607194041815, 0.9522728522471394, 0.37549107236304946, 0.3779483833135727, 0.9133919702694991, 0.9583904722626736, 0.6860379726044363, 0.19352355780667013, 0.2751650172095135,
              0.17004291764742357, 0.29761719449391844, 0.4439185162973016, 0.6047872401880852, 0.6085953952893494, 0.22712363588512985, 0.9595173052322611, 0.7040422998860274, 0.7787252838258393, 0.44996002837615867, 0.25191360600884943, 0.18915006378179716, 0.22531691981445015, 0.19326378643132014, 0.4684096266178418, 0.906399796477089, 0.11765581197252967, 0.782308677379079, 0.45757934371778664, 0.7979233704136902, 0.6414935426833389, 0.8420692954503183, 0.3895765784590167, 0.7585778904787359, 0.10455903741922691, 0.6206624732023196, 0.9796298201506508, 0.9606177167415177, 0.5179399999434361, 0.15762716938876886, 0.8676373793561578, 0.5427633193086049, 0.4799303415357836, 0.2033111906206814, 0.6688946296743229, 0.7047813416795993, 0.4523019758288376, 0.3088516239649981, 0.11967411651870888, 0.4688505677674346, 0.5021931141971973, 0.8268410278729638, 0.10560781024786377, 0.17450656988456134, 0.7488159553747705, 0.162886660035095, 0.49223779537770773, 0.6946316349715498, 0.866433417483596, 0.4987274972154929]

    Cg_min = [0.7727245999513712, 0.24543375900825748, 0.785617489166371, 0.5658868104894168, 0.8784816828704456, 0.13796072143418978, 0.28189000743276094, 0.7528327054082473, 0.5431823065612567, 0.2846416658825203, 0.47007151495541843, 0.8919067112209303, 0.5399916816994709, 0.9212945115051006, 0.4900825795039112, 0.1728213780421985, 0.18027192084958119, 0.9601710824556567, 0.47560403024969267, 0.9941833808555566, 0.3367510416996713, 0.757246381385742, 0.9404388258477961, 0.3932017131739377, 0.7015272960437607, 0.41178813650420887, 0.803915962519426, 0.22714845908778483, 0.2956571665937919, 0.10042322161584305, 0.9829224441586885, 0.6583455728032721, 0.3129465962550104, 0.9413621357765016, 0.34913868047788843, 0.9605197903862901, 0.9340088947904703, 0.9696566833754489, 0.2144087630490015, 0.48343029432430284, 0.973631584793158, 0.6821517011011357, 0.9002639058714675, 0.6781625071336378, 0.8971105376120788, 0.14804883075339162, 0.5043274121669227, 0.3964626259193075, 0.6304352633948127, 0.5769118105986385,
              0.8455972819684169, 0.4329615794190508, 0.33673556826647066, 0.5300534984258775, 0.5307378849275272, 0.26801856441071337, 0.9327857782978386, 0.5905984448096933, 0.43630410301869194, 0.3144090833689158, 0.5151072277723872, 0.22161617512700607, 0.5342778418490036, 0.5716624464659544, 0.592038511189694, 0.9117439170635968, 0.8948469530007375, 0.2457229093504071, 0.3992775542079383, 0.4349828243724332, 0.17904437454591343, 0.27629136443081914, 0.915318755438518, 0.7481059926835975, 0.8321023066579128, 0.8576854354449804, 0.6892385699289374, 0.7719946386395943, 0.9602480023008727, 0.13565368709649867, 0.19813756131709642, 0.6166632396092872, 0.22632380672721977, 0.7757482434111332, 0.2170544889969569, 0.8091564670384911, 0.14873371644490913, 0.5483845374273648, 0.2707943900478642, 0.6628945968108156, 0.9113426089978361, 0.4864247234536996, 0.1110587788786339, 0.6490380192285795, 0.7860335286624619, 0.42663248774772156, 0.8806250495947053, 0.2534338758236515, 0.1255172545572136, 0.7315119979802447]

    Cp_max = [2.8980666166046953, 1.6206890344183884, 3.2145841800607746, 1.7811865005178271, 1.7853448438348503, 2.191936628818026, 1.555357542231143, 3.5941712638764027, 2.2963818450179003, 1.8444850868438145, 3.328279228299863, 2.4197276697850025, 3.2369040956174544, 2.411401961309182, 1.982489728131591, 1.7152575645592238, 2.603142776166658, 2.2534801332215952, 2.3235353649118724, 3.345064381264214, 3.2007459304156347, 3.081039656192443, 2.865737156866084, 2.94926583119412, 2.525742999875172, 3.79636528306107, 3.6287384750459957, 3.2435173250200586, 2.9821137615136664, 2.736199545502293, 2.186469210861123, 1.6658776921975011, 3.7932200409164984, 1.5389022341103475, 3.539401019496529, 2.1942357223162747, 3.4707908260071507, 3.8895717588450935, 3.4659325165280697, 2.9826305830729156, 3.920358581721492, 3.202572875352478, 2.5913228452651262, 3.630584863036878, 2.3435209657030383, 3.962932330878638, 3.4624950976995192, 3.0980640926409517, 3.2435640610789296, 3.7289468387442977,
              2.3111756663568057, 2.99481142439907, 2.9597502959354394, 2.175456095116634, 1.5754065578092642, 3.9773084932102747, 2.207401371140876, 2.1983823905932796, 3.423701731060934, 3.8265496043039318, 1.571038044136242, 2.412023685426228, 1.8121918997234308, 3.0386715804313917, 3.5920926888385933, 3.1219331464372955, 1.964217189055032, 2.7266380056809876, 1.8624768317184919, 1.6982708177881778, 2.796396466451595, 1.5003457206005055, 3.9083527982289588, 3.1503029605359307, 2.1812067575634315, 3.381581897105857, 1.699680854461596, 3.849113477935996, 3.0935856996551148, 3.208688806324198, 2.1872004400737177, 3.841582338218716, 2.8432564691587356, 3.5773322567044357, 3.5189241053684617, 2.8444307463276317, 1.9746399841740083, 2.1018284051824745, 2.0275116551395422, 3.0852916523273652, 2.5636721722892144, 3.0629079836195716, 2.7046846963429836, 3.486325991488343, 1.6801553838586878, 1.8688516225941005, 2.705723438903796, 1.7688952271636098, 3.0162716215554695, 3.801741661733433]

    Cg_max = [2.8883640995914184, 3.522856466079572, 2.283417689452805, 2.6870904859553963, 2.936280340919189, 2.4446864738923244, 2.6971503841493933, 3.4020017800985576, 3.410175833154958, 2.3851152776490885, 1.537788247181624, 2.969870652822472, 3.587179170044386, 1.6592171762256256, 3.199768254720707, 2.940102283395162, 1.8963496263581359, 2.4751222848697205, 1.6545944331313138, 3.930487238276145, 3.0144058193073664, 2.3360502080513887, 3.1215348971679884, 1.9745222420043247, 2.1269234706396407, 3.6544802633399134, 3.390303221467031, 1.7022251495570437, 1.7895260147041032, 2.978585134318526, 3.9009723062100066, 1.5158484076529404, 2.6289045118052865, 3.479317918047916, 3.895789161401207, 1.7233016709893907, 3.5799624617086887, 2.176804542386139, 3.702861064255594, 3.7345401777091602, 2.3835456496287475, 2.1945585691960465, 3.390008167140031, 3.070407365306239, 1.5046300332177154, 2.5263390991315826, 1.944798690466802, 3.7254966613665257, 3.03026645287984, 3.304496721253697,
              3.1676848285633694, 3.727765394319377, 2.2056015656308574, 3.2717144351936382, 3.418406215412533, 3.7951828186351912, 3.514566161704549, 2.2212597880561096, 2.8006127297255645, 3.159279230599638, 2.65773434930342, 2.202845488987007, 2.6568825841398134, 2.7898619953712283, 2.7647165903239967, 2.5268496466035706, 2.376927919413216, 2.9428760643735252, 3.0962746861283446, 3.0150653258593816, 1.9010106588662354, 3.482304004796766, 2.4471075344953044, 3.3650353448322847, 3.7980322818858188, 2.5282668052729345, 1.9303891417617072, 3.3596848421619523, 3.4968063545652965, 1.8094252693884987, 1.6766374124060879, 1.7766028872346689, 3.2856311543497583, 3.6079835361584616, 3.935494261490175, 1.6532196197616447, 2.575021424223965, 2.5762803351446752, 2.015518936781678, 2.3509614106098065, 3.9300488597814742, 2.7250419760033893, 2.0840853030888837, 2.4872292134040617, 3.5645430904820703, 1.5340273811614784, 1.5003581183916965, 3.988099962850725, 2.6148444456211024, 3.502317927941579]

    w_min = [0.7160605693467496, 0.7993218895317963, 0.7683963781583575, 0.7047530390385035, 0.586330122945824, 0.09367907542924611, 0.8717178007991856, 0.941965322001564, 0.6689472535491467, 0.2688144658517782, 0.4103548419017885, 0.5769797704721505, 0.9190327613652907, 0.4898891209062801, 0.2705123845696286, 0.2679211238584719, 0.2682502131528513, 0.08805666792082266, 0.828212726153669, 0.7269219958360611, 0.5482477274494549, 0.2250133734355817, 0.2665330484356371, 0.8612968337095651, 0.8103872698363291, 0.5883117348170457, 0.11221056491942563, 0.05583809526560132, 0.269675944697879, 0.8875747664846424, 0.8495068093313561, 0.5304004112365951, 0.7208644192437896, 0.2887830226630158, 0.365450710806936, 0.1998827397181297, 0.14883425595269578, 0.41391122922374246, 0.27558019710681925, 0.5528068664998117, 0.7600483625594291, 0.42779354715438483, 0.5672930878539275, 0.8694399943840502, 0.25640474442382216, 0.4716172872628589, 0.4173330931945804, 0.8196026438956188, 0.22203435862573717, 0.6679382287273756,
             0.8482420381457477, 0.5422323396203214, 0.5000004485964644, 0.412145040272656, 0.8544169332070023, 0.43280360196549844, 0.6878793606463113, 0.8197986224123437, 0.8753892951909545, 0.6903930698424464, 0.743648755355766, 0.32361085922245386, 0.10319254411236306, 0.3561687437351463, 0.3734215152199651, 0.9288942378078501, 0.3289172302789386, 0.5054523827890837, 0.4645312681815695, 0.9382553299596462, 0.8207216477238876, 0.07703156007266224, 0.957733433344821, 0.8085006709796928, 0.6886764192992293, 0.6093615210589779, 0.8573159795380701, 0.5246441321660469, 0.8355732500059176, 0.21570180895301672, 0.9932491731398212, 0.2978954472894768, 0.14288201623701902, 0.8408149013841508, 0.8472244742355254, 0.9946918761415792, 0.5167301654873211, 0.8946125048108404, 0.24902633002187402, 0.12500748117345356, 0.7993557553963062, 0.5106137253284987, 0.7780178546493673, 0.5164404258934637, 0.9462499036705445, 0.9455147637568251, 0.2865600017421221, 0.287632492477365, 0.24555077401679748, 0.24008288139065503]

    w_max = [1.635114752939525, 1.641447993870254, 1.1120254237588618, 0.7986392160964976, 1.1480536658465317, 0.8049984118971738, 0.8239163263278237, 1.8002828565849223, 0.9780829155937356, 1.8084604521076595, 0.5920822290421839, 1.9857607292388162, 1.6239328014886087, 0.8318597104634914, 0.6777227754509025, 0.9483554087502268, 0.5808119195190748, 1.3486705746461007, 1.4896317793957448, 1.2718221862070462, 0.5579700478729688, 1.5745961543982396, 1.1249030311340267, 1.2199892541988666, 1.1433758382002013, 1.6058098199448423, 1.8417263659088785, 0.5940004882268552, 1.4014515560168617, 0.7711030982497133, 0.5439537093340483, 1.4811828221006966, 1.324390589334674, 0.5481855493915297, 0.5839135920597689, 0.5371444614494137, 0.8532617812949094, 0.7802694317056738, 0.6936799921661079, 1.8550395685438954, 1.1364910548421943, 1.2511438770118142, 1.5633258944197883, 1.092867029244003, 0.5247122424608959, 1.2867437470256884, 1.7231701788333567, 1.6373863804659543, 1.2489921922874536,
             1.571943402934191, 1.446527532561058, 0.905080232920257, 1.0598259738124916, 1.7900366925765665, 0.8750996937353259, 1.4052801021024877, 0.5094635752227392, 0.9683249749404604, 1.5822200564807525, 1.8309851927490641, 0.7290064644997631, 1.691824167779599, 0.73089173979908, 1.1868166219540472, 0.766299067322637, 1.7815895729903128, 0.5357688636599858, 0.5992253317148992, 0.891575249760821, 1.4715685958669449, 1.4575987024486927, 0.8563491938325807, 1.833659441424238, 1.8715668771950278, 1.8276633687196238, 1.2306902583876946, 1.9970716315182648, 1.3449232118353567, 1.207695201793964, 1.803418811388156, 1.4009990449299492, 1.5845377192505372, 1.0140098522799428, 0.6775261347561576, 0.839464825646576, 1.330346044497888, 0.8855646333314073, 1.7493955962747516, 0.562035939643597, 1.5795883095720178, 1.2427774260940703, 0.5091791902052323, 0.5610336645164687, 1.2221580507878, 1.916675815246648, 1.596726156329619, 1.271924256001701, 0.9340827078589008, 1.3756473456448246, 1.471610628085738]

    gwn_std_dev = [0.149854227462725, 0.09846288381784714, 0.12259878685574953, 0.15271033047975013, 0.17711454512944588, 0.2278258055350056, 0.16870001074588156, 0.24755367712540935, 0.28216954992058935, 0.07423978341526306, 0.07966891450824376, 0.28834936840895053, 0.12268091602946363, 0.022323132583725452, 0.24369057027501922, 0.2649555159230909, 0.10852189097111518, 0.10578087650537732, 0.09332733946600039, 0.17087125464309974, 0.024400112684847472, 0.011851259677496838, 0.2743122199188355, 0.18601211671435017, 0.12051782843569818, 0.28373898389350105, 0.1360084389092428, 0.12383693767371029, 0.27860509608040196, 0.12338402121309629, 0.012475435850395603, 0.10413252518370315, 0.11603247220361176, 0.297851587640781, 0.2514088860519822, 0.08169340620001109, 0.0962261042925226, 0.08995081456700636, 0.2524881694750808, 0.22102708815809355, 0.15372449380578504, 0.15927343361980023, 0.18459573574311702, 0.29931668107953274, 0.13630049391756277, 0.16077949880330855, 0.22479957622105906, 0.2871427968850316, 0.13789243541062368, 0.13296739448397843,
                   0.2091154985348382, 0.07220347661820516, 0.26665192385644576, 0.26676411182120086, 0.23177250335200084, 0.044521017743890536, 0.11904620152473877, 0.21763573638394723, 0.2546491976892733, 0.13244619413536102, 0.1711145196960935, 0.08700982117713667, 0.11382746372481554, 0.1884115557362195, 0.13729164504259242, 0.2954897242400484, 0.09669361686912037, 0.2522727416812527, 0.2789695291959802, 0.14099131635176404, 0.028170516135559896, 0.27057857503516985, 0.21970428973954964, 0.22311844839396727, 0.05617752478981964, 0.1775517733495292, 0.07750091091441681, 0.04396579017650296, 0.2883468584507717, 0.11193668209762955, 0.20388753443046456, 0.27585192842157913, 0.11668510348058565, 0.12162189518342756, 0.11300126818250482, 0.24846077205669018, 0.17596973033960012, 0.02586081737450012, 0.12556769800863474, 0.18270568186841252, 0.15508765130421898, 0.1833845158861564, 0.1591048147090378, 0.28575142816357085, 0.01670515113859424, 0.018007470824330685, 0.29926592270207925, 0.18535378463655164, 0.14109280147218983, 0.026968749021956978]

    parameter_dicts = {}

    for i in range(len(Cp_min)):
        parameter_dicts[i] = {
            "velocity_bounds": velocity_bounds[i],
            'Cp_min': Cp_min[i],
            'Cp_max': Cp_max[i],
            'Cg_min': Cg_min[i],
            'Cg_max': Cg_max[i],
            'w_min': w_min[i],
            'w_max': w_max[i],
            'gwn_std_dev': gwn_std_dev[i]
        }

        run_pso_partial = partial(
            run_pso,
            iterations=iterations,
            position_bounds=position_bounds,
            velocity_bounds=parameter_dicts[i]["velocity_bounds"],
            fitness_threshold=fitness_threshold,
            num_particles=num_particles,
            Cp_min=parameter_dicts[i]["Cp_min"],
            Cp_max=parameter_dicts[i]["Cp_max"],
            Cg_min=parameter_dicts[i]["Cg_min"],
            Cg_max=parameter_dicts[i]["Cg_max"],
            w_min=parameter_dicts[i]["w_min"],
            w_max=parameter_dicts[i]["w_max"],
            gwn_std_dev=parameter_dicts[i]["gwn_std_dev"],
        )

        pso_runs = multiprocessing.cpu_count() - 1

        start_time = time.time()
        with multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1) as pool:
            results = pool.map(run_pso_partial, range(pso_runs))
        end_time = time.time()
        elapsed_time = end_time - start_time

        fitness_histories = [result[2] for result in results]
        (
            swarm_best_fitness,
            swarm_best_position,
            swarm_fitness_history,
            swarm_position_history,
        ) = zip(*results)
        mean_best_fitness = np.mean(swarm_best_fitness)
        min_best_fitness = np.min(swarm_best_fitness)
        max_best_fitness = np.max(swarm_best_fitness)

        index_of_best_fitness = swarm_best_fitness.index(min_best_fitness)
        best_weights = swarm_best_position[index_of_best_fitness].tolist()

        sys.stdout.write(
            f"Minimum fitness for {pso_runs} runs: {min_best_fitness}. Mean: {mean_best_fitness}. Max: {max_best_fitness}"
        )

        save_opt_ann_rpso_stats(
            fitness_histories,
            "rpso",
            pso_runs,
            position_bounds,
            parameter_dicts[i]["velocity_bounds"],
            fitness_threshold,
            num_particles,
            parameter_dicts[i]["Cp_min"],
            parameter_dicts[i]["Cp_max"],
            parameter_dicts[i]["Cg_min"],
            parameter_dicts[i]["Cg_max"],
            parameter_dicts[i]["w_min"],
            parameter_dicts[i]["w_max"],
            parameter_dicts[i]["gwn_std_dev"],
            iterations,
            elapsed_time,
            min_best_fitness,
            mean_best_fitness,
            max_best_fitness,
            best_weights,
        )

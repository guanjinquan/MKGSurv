from modules.base_modules.panther_module import StructuredPANTHER
try:
    from modules.base_modules.trans_mil_module import AggregatingTransMIL
except ModuleNotFoundError:
    AggregatingTransMIL = None




def GetImageAggregater(image_aggregater, InputDim, OutputDim, OutputTokenNum, PrototypesData=None):

    if image_aggregater == 'transmil':
        if AggregatingTransMIL is None:
            raise ImportError("AggregatingTransMIL requires the optional nystrom_attention package.")
        return AggregatingTransMIL(
            input_dim=InputDim,
            embed_dim=OutputDim,
            num_aggregated_tokens=OutputTokenNum
        )

    elif image_aggregater == 'panther':
        assert PrototypesData.shape[0] == OutputTokenNum, f"Expect {OutputTokenNum} == {PrototypesData.shape[0]}"
        return StructuredPANTHER(
            in_dim=InputDim,
            out_dim=OutputDim,
            n_proto=OutputTokenNum,
            prototypes=PrototypesData,
        )


    raise ValueError(f"Not Support Model: {image_aggregater}")


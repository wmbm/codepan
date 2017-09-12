MODEL_PARAMS = \
{
    "aggregationInfo": {
        "seconds": 0, 
        "fields": [], 
        "months": 0, 
        "days": 0, 
        "years": 0, 
        "hours": 0, 
        "microseconds": 0, 
        "weeks": 0, 
        "minutes": 0, 
        "milliseconds": 0
    }, 
    "model": "HTMPrediction", 
    "version": 1, 
    "predictAheadTime": None, 
    "modelParams": {
        "sensorParams": {
            "sensorAutoReset": None, 
            "encoders": {
                "timestamp_dayOfWeek": None, 
                "timestamp_timeOfDay": {
                    "fieldname": "timestamp", 
                    "timeOfDay": [
                        21, 
                        1
                    ], 
                    "type": "DateEncoder", 
                    "name": "timestamp_timeOfDay"
                }, 
                "value": {
                    "name": "consumption", 
                    "resolution": 1, 
                    "seed": 2048, 
                    "fieldname": "consumption", 
                    "type": "RandomDistributedScalarEncoder"
                }, 
                "timestamp_weekend": {
                    "fieldname": "timestamp", 
                    "weekend": 21, 
                    "type": "DateEncoder", 
                    "name": "timestamp_weekend"
                }, 
            }, 
            "verbosity": 0
        }, 
        "anomalyParams": {
            "anomalyCacheRecords": None, 
            "autoDetectThreshold": None, 
            "autoDetectWaitRecords": 5030
        }, 
        "spParams": {
            "columnCount": 2048, 
            "synPermInactiveDec": 0.0005, 
            "spatialImp": "cpp", 
            "inputWidth": 512, 
            "spVerbosity": 0, 
            "synPermConnected": 0.2, 
            "synPermActiveInc": 0.003, 
            "potentialPct": 0.8, 
            "numActiveColumnsPerInhArea": 40, 
            "boostStrength": 0.0, 
            "globalInhibition": 1, 
            "seed": 1956
        }, 
        "trainSPNetOnlyIfRequested": False, 
        "clParams": {
            "alpha": 0.035828933612158, 
            "verbosity": 0, 
            "steps": "1", 
            "regionName": "SDRClassifierRegion"
        }, 
        "tmParams": {
            "columnCount": 2048, 
            "activationThreshold": 13, 
            "pamLength": 3, 
            "cellsPerColumn": 32, 
            "permanenceDec": 0.1, 
            "minThreshold": 10, 
            "inputWidth": 2048, 
            "maxSynapsesPerSegment": 32, 
            "outputType": "normal", 
            "initialPerm": 0.21, 
            "globalDecay": 0.0, 
            "maxAge": 0, 
            "newSynapseCount": 20, 
            "maxSegmentsPerCell": 128, 
            "permanenceInc": 0.1, 
            "temporalImp": "cpp", 
            "seed": 1960, 
            "verbosity": 0
        }, 
        "clEnable": True, 
        "spEnable": True, 
        "inferenceType": "TemporalAnomaly", 
        "tmEnable": True
    }
}

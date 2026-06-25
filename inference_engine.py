def infer_mission(detected_components):
    if not detected_components:
        return {
            "mission_type": "Unknown Object",
            "reasoning": "The visual engine could not identify any recognizable aerospace components.",
            "confidence_note": "Inference aborted due to insufficient visual data.",
            "component_explanations": {}
        }

    # Clean up long raw labels into exact, unified strings
    components = {}
    for k, v in detected_components.items():
        clean_key = k.replace("a ", "").replace("or propulsion nozzles", "").replace("or optical sensor", "").strip()
        components[clean_key] = v

    # --- SAFETY CALIBRATION FILTER ---
    # If background noise is detected and holds a significant score, trigger rejection
    if "background noise" in components:
        bg_score = components["background noise"]
        # If background noise is the strongest detection, reject immediately
        is_highest = all(bg_score >= val for val in components.values())
        
        if is_highest:
            return {
                "mission_type": "Non-Aerospace Object",
                "reasoning": "The telemetry profile matches ambient background textures or everyday earthbound structures rather than verified spacecraft configurations.",
                "confidence_note": f"Calibration Engine threshold triggered ({bg_score}% background match).",
                "component_explanations": {}
            }

    # Human-friendly descriptions using the exact same clean keys
    descriptions = {
        "solar panels": "Generates electrical power from sunlight to keep the onboard electronics running indefinitely.",
        "parabolic satellite dish antenna": "Focuses and transmits radio signals back to Earth or other spacecraft over long distances.",
        "camera lens": "Captures high-resolution imagery or tracks stellar objects for scientific observations.",
        "rocket thrusters": "Provides directional force (thrust) for maneuvers, orbital corrections, or deep space propulsion."
    }

    component_explanations = {}
    for comp in components:
        if comp in descriptions:
            component_explanations[comp] = descriptions[comp]

    # --- INFERENCE RULES LOGIC ---
    mission_type = "Generic/Unknown Spacecraft"
    reasoning = "Not enough specific components detected to isolate a concrete mission profile."
    confidence = "Low"

    # Remove background noise from structural inference rules if it didn't win outright
    active_space_parts = {k: v for k, v in components.items() if k != "background noise"}

    if "camera lens" in active_space_parts:
        mission_type = "Earth Observation / Scientific Survey"
        reasoning = "The presence of optical sensors suggests the craft is designed to capture imagery or environmental data."
        confidence = "High" if len(active_space_parts) > 1 else "Medium"

    elif "parabolic satellite dish antenna" in active_space_parts and "solar panels" in active_space_parts:
        mission_type = "Commercial Communications Satellite"
        reasoning = "The combination of high-power solar arrays and high-gain antennas is typical for data relay or broadcasting missions."
        confidence = "High"

    elif "rocket thrusters" in active_space_parts:
        mission_type = "Deep Space Probe or Rocket Propulsion Stage"
        reasoning = "Prominent thrusters point toward heavy maneuvering, interplanetary travel, or orbital insertion tasks."
        confidence = "Medium"

    return {
        "mission_type": mission_type,
        "reasoning": reasoning,
        "confidence_note": f"Confidence is {confidence} based on available visual cues.",
        "component_explanations": component_explanations
    }
import random
import re
from utils import InstType, Location, Topic

class Instructions:
    def __init__(self, seed = None, inst_type_weights={}, location_weights={}, topic_weights={}):
        self.inst_types = list(InstType)
        self.locations = list(Location)
        self.topics = list(Topic)

        self.inst_type_weights = inst_type_weights if len(inst_type_weights) else {inst_type: 1 for inst_type in InstType}
        self.location_weights = location_weights if len(location_weights) else {loc: 1 for loc in Location}
        self.topic_weights = topic_weights if len(topic_weights) else {top: 1 for top in Topic}

        self.inst_list = self._make_inst()

        self.seed = seed
    
    def set_weights(self, inst_type_weights = {}, location_weights={}, topic_weights={}):

        if inst_type_weights:
            for k, v in inst_type_weights.items():
                assert k in self.inst_type
                self.inst_type_weights[k] = v

        if location_weights:
            for k, v in location_weights.items():
                assert k in self.locations
                self.location_weights[k] = v

        if topic_weights:
            for k, v in topic_weights.items():
                assert k in self.topics
                self.topic_weights[k] = v


    def _make_inst(self):

        instructions = {}

        for inst_type in self.inst_types:
            instructions[inst_type] = {}

            for location in self.locations:
                instructions[inst_type][location] = []

                for topic in self.topics:
                    include = False

                    if (location in {Location.LINE_FIRST, Location.LINE_LAST} and inst_type != InstType.ADD_OBS): 
                        include = False

                    elif inst_type == InstType.ADD_OBS:
                        include = True 

                    elif inst_type in {InstType.CHG_OBS, InstType.RM_OBS}:
                        include = True

                    elif (inst_type == InstType.CHG_LOC_OF_OBS and topic not in {
                        Topic.OPACITY_DIFFUSE_EDEMA, Topic.LUNG_HYPERINFLATION, Topic.MEDIASTINAL_ENLARGEMENT,
                        Topic.MEDIASTINAL_ECTATIC_AORTA, Topic.MEDIASTINAL_CALCIFIED_AORTA, Topic.CARDIAC_CARDIOMEGALY,
                        Topic.CARDIAC_PNEUMOPERICARDIUM, Topic.CARDIAC_PERICARDIAL_EFFUSION, Topic.HILAR_ENLARGEMENT,
                        }):
                        include = True

                    elif (inst_type == InstType.CHG_SHAPE_OF_OBS and topic in {
                        Topic.OPACITY_CONSOLIDATION, Topic.OPACITY_LESION, Topic.OPACITY_NODULES,  
                        Topic.OPACITY_MASS, Topic.PLEURAL_ABNORMALITY, Topic.PLEURAL_PNEUMOTHORAX,
                        Topic.PLEURAL_EFFUSION, Topic.BONE_RIB_FRACTURE, 
                        } and location in {Location.REPORT, Location.SECTION, Location.SECTION_FIND, Location.SECTION_IMPR}):
                        include = True

                    elif (inst_type in {InstType.CHG_SEVERITY} and 
                          topic not in {
                            Topic.FOREIGN_BUTTON, Topic.FOREIGN_HAIR, Topic.FOREIGN_GRAVEL, Topic.FOREIGN_NEEDLE, 
                            Topic.FOREIGN_BULLET, Topic.DEVICE_TUBE, Topic.DEVICE_FEEDING_TUBE, Topic.DEVICE_FEEDING_TUBE,
                            Topic.DEVICE_CHEST_TUBE, Topic.DEVICE_PACEMAKER, Topic.DEVICE_ICD, Topic.DEVICE_LVAD,
                            Topic.DEVICE_ORTHOPEDIC_WIRE, Topic.DEVICE_ORTHOPEDIC_PLATE, Topic.DEVICE_ORTHOPEDIC_SCREW,
                            Topic.DEVICE_ORTHOPEDIC_NAIL, Topic.DEVICE_SURGICAL_STAPLES} and 
                          location in {Location.REPORT, Location.SECTION, Location.SECTION_FIND, Location.SECTION_IMPR}):
                        include = True

                    elif (inst_type == InstType.CHG_CERTAINTY and topic in {
                        Topic.OPACITY_ATELECTASIS, Topic.OPACITY_NODULES, Topic.PLEURAL_EFFUSION}):
                        include = True

                    elif (inst_type in {InstType.ADD_REC, InstType.RM_REC, InstType.CHG_REC} and 
                          topic in {Topic.OPACITY_LESION, Topic.OPACITY_NODULES, Topic.OPACITY_MASS
                                    } and 
                          location in {Location.REPORT, Location.SECTION, Location.SECTION_IMPR}):
                        include = True

                    elif (inst_type in {InstType.ADD_COMPS_TO_PRIOR}): 
                        include= True

                    elif (inst_type in {InstType.RM_COMPS_TO_PRIOR} ): 
                        include= True

                    if include:
                        instructions[inst_type][location].append(topic)
                        # instructions.append({
                        #     '': inst_type,
                        #     'level': location,
                        #     'topic': topic
                        # })

        return instructions

    def get_all_inst(self):
        return self.inst_list   
    
    def search_report(self, report, topic, section=""):
        if not section: 
            return topic.lower() in report.lower()

        findings, impression = report.split('FINDINGS:')

        if 'f' in section.lower():
            return topic.lower() in findings.lower()

        return topic.lower() in impression.lower()


    def get_single_inst(self, report):

        if self.seed:
            random.seed(self.seed)

        found = False
        tries = 0 
        res = ""

        while not found and tries < 100:
            tries += 1
            
            # randomly choose instruction type
            inst_type = random.choices(
                self.inst_types, 
                weights=[self.inst_type_weights[i] for i in self.inst_types], 
                k=1)[0]

            # get possible locations
            locations = list(self.inst_list.get(inst_type).keys())

            if not locations:
                continue 

            # randomly choose location
            location = random.choices(
                locations, 
                weights=[self.location_weights[loc] for loc in locations], 
                k=1)[0]

            if inst_type not in {
                InstType.ADD_COMPS_TO_PRIOR, InstType.ADD_OBS, InstType.ADD_REC
            }:
                if location == Location.SECTION_FIND and "findings:" not in report.lower():
                    location = Location.SECTION_IMPR
                elif location == Location.SECTION_IMPR and "impression:" not in report.lower():
                    location = Location.SECTION_FIND

            # for certain instructions, check existence of topics
            if inst_type in {InstType.RM_OBS, InstType.CHG_OBS, InstType.CHG_LOC_OF_OBS, 
                             InstType.CHG_SHAPE_OF_OBS, InstType.CHG_CERTAINTY, InstType.CHG_SEVERITY}:

                res = f"an instruction to {inst_type.value} {location.value}" 
                found = True

            elif (inst_type in {InstType.CHG_COMPS_TO_PRIOR, InstType.RM_COMPS_TO_PRIOR, 
                                InstType.RM_REC, InstType.CHG_REC}):
                res = f"an instruction to {inst_type.value} {location.value}, if any" 
                found = True

            elif inst_type in { InstType.ADD_REC, InstType.ADD_COMPS_TO_PRIOR}:
                topics = self.inst_list.get(inst_type).get(location)
                if topics:
                    topic = random.choices(
                        topics, 
                        weights=[self.topic_weights[t] for t in topics], 
                        k=1)[0]
                    found = True

            elif inst_type in {InstType.ADD_OBS} :
                topics = self.inst_list.get(inst_type).get(location)
                if topics:
                    topic = random.choices(
                        topics, 
                        weights=[self.topic_weights[t] for t in topics], 
                        k=1)[0]
                    found= True
        
                # topics_sub = []
                # for t in topics:
                #     if location in {Location.SECTION_FIND, Location.SECTION_IMPR}:
                #         mentioned = self.search_report(report, t, location.value)
                #     else:
                #         mentioned = self.search_report(report, t)

                #     if mentioned:
                #         topics_sub.append(t)

                # if topics_sub: 
                #     found = True
                #     topics = topics_sub


        if tries == 100 and not found:
            return None

        if not res: 
            assert (inst_type in self.inst_types and location in self.locations 
                    and topic in self.topics) 
            res = f"an instruction to {inst_type.value} {location.value} about {topic.value}" 
        return res

    def get_insts(self, report, n=1, inst_types=[], levels=[], topics=[]):

        included = []
        for _ in range(n):
            inst = self.get_single_inst(report)
            if inst and inst not in included:
                included.append(inst) 
        
        if len(included) > 0:
            return "Create the following instruction(s): " + '; '.join(included) 
        return None

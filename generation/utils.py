from enum import Enum
from openai import AzureOpenAI
from torch.utils.data import Dataset

class InstType(Enum):
    ADD_OBS           = "add an observation"
    RM_OBS            = "remove an observation"
    CHG_OBS           = "change an observation"
    CHG_LOC_OF_OBS    = "change the anatomical location of an observation"
    CHG_SHAPE_OF_OBS  = "change the anatomical shape or size of an observation"
    CHG_SEVERITY      = "change the severity of an observation"
    CHG_CERTAINTY     = "change the certainty of an observation"
    ADD_REC           = "add a recommendation"
    RM_REC            = "remove a recommendation"
    CHG_REC           = "change a recommendation"
    ADD_COMPS_TO_PRIOR= "add comparisons to prior"
    RM_COMPS_TO_PRIOR = "remove comparisons to prior"
    CHG_COMPS_TO_PRIOR= "change comparisons to prior"

class Location(Enum):
    REPORT            = "in the entire report"
    SECTION           = "in the FINDINGS or IMPRESSION section"
    SECTION_FIND      = "in the FINDINGS section"
    SECTION_IMPR      = "in the IMPRESSION section"
    LINE              = "of a line"
    LINE_FIRST        = "of the first line"
    LINE_LAST         = "of the last line"

class Topic(Enum):

    OPACITY_EDEMA                       = "edema"
    OPACITY_DIFFUSE_EDEMA               = "diffuse edema"
    OPACITY_PATCHY_EDEMA                = "patchy edema"
    OPACITY_CONSOLIDATION               = "consolidation"
    OPACITY_ATELECTASIS                 = "atelectasis"
    OPACITY_LESION                      = "lesion"
    OPACITY_NODULES                     = "nodule"
    OPACITY_MASS                        = "mass"
    OPACITY_CYST                        = "cyst/bulla/bleb"

    LUNG_HYPERINFLATION                 = "lung hyperinflation"
    LUNG_COPD                           = "Emphysema/COPD"

    PLEURAL_ABNORMALITY                 = "pleural abnormality"
    PLEURAL_PNEUMOTHORAX                = "pneumothorax"
    PLEURAL_TENSION_PNEUMOTHORAX        = "tension pneumothorax"
    PLEURAL_NON_TENSION_PNEUMOTHORAX    = "non-tension pneumothorax"
    PLEURAL_EFFUSION                    = "pleural effusion"
    PLEURAL_FREEFLOW_EFFUSION           = "free-flowing effusion"
    PLEURAL_LOCULATED_EFFUSION          = "loculated effusion"
    PLEURAL_THICKENING                  = "pleural thickening"
    PLEURAL_PLEURAL_MASS                = "pleural mass"
    PLEURAL_SCARRING                    = "pleural scarring"
    PLEURAL_PLAQUE                      = "pleural plaque"

    MEDIASTINAL_ENLARGEMENT             = "mediastinal enlargement"
    MEDIASTINAL_PHEUMO                  = "pneumomediastinum"
    MEDIASTINAL_ECTATIC_AORTA           = "ectatic aorta"
    MEDIASTINAL_CALCIFIED_AORTA         = "calcified aorta"
    CARDIAC_CARDIOMEGALY                = "cardiomegaly"
    CARDIAC_PNEUMOPERICARDIUM           = "pneumopericardium"
    CARDIAC_PERICARDIAL_EFFUSION        = "pericardial effusion"
    HILAR_ENLARGEMENT                   = "hilar enlargement"
    PULMONARY_ENLARGEMENT               = "enlarged pulmonary"

    CHEST_WALL_SUBCUTANEOUS             = "subcutaneous emphysema"
    CHEST_WALL_SPINE                    = "spine degenration"

    BONE_RIB_FRACTURE                   = "rib fracture"
    BONE_SPINE_DEGENERATIVE             = "degenerative changes in spine"
    BONE_SPINE_SCOLIOSIS                = "scoliosis"
    BONE_SPINE_COMPRESSION_FRACTURE     = "spine compression fracture"
    BONE_SHOULDER_DEGENERATIVE          = "degenerative changes in shoulder"
    BONE_SHOULDER_DISLOCATION           = "shoulder dislocation"
    BONE_SHOULDER_FRACTURE              = "shoulder fracture"

    FOREIGN_BUTTON                      = "button"
    FOREIGN_HAIR                        = "hair"
    FOREIGN_GRAVEL                      = "gravel"
    FOREIGN_NEEDLE                      = "needle"
    FOREIGN_BULLET                      = "bullet"

    DEVICE_TUBE                         = "tube"
    DEVICE_FEEDING_TUBE                 = "feeding tube"
    DEVICE_ENDOTRACHEAL                 = "endotracheal tube"
    DEVICE_CHEST_TUBE                   = "chest tube"
    DEVICE_PACEMAKER                    = "pacemaker"
    DEVICE_CATHETER                     = "catheter"
    DEVICE_ICD                          = "ICD"
    DEVICE_LVAD                         = "LVAD"
    DEVICE_ORTHOPEDIC_WIRE              = "orthopedic wire"
    DEVICE_ORTHOPEDIC_PLATE             = "orthopedic plate"
    DEVICE_ORTHOPEDIC_SCREW             = "orthopedic screw"
    DEVICE_ORTHOPEDIC_NAIL              = "orthopedic nail"
    DEVICE_SURGICAL_STAPLES             = "surgical staples"


topic_variants = {
    Topic.OPACITY_EDEMA: ["edema", "fluid buildup"],
    Topic.OPACITY_DIFFUSE_EDEMA: ["diffuse edema", "widespread fluid buildup"],
    Topic.OPACITY_PATCHY_EDEMA: ["patchy edema", "localized fluid buildup"],
    Topic.OPACITY_CONSOLIDATION: ["consolidation"],
    Topic.OPACITY_ATELECTASIS: ["atelectasis", "collapsed lung"],
    Topic.OPACITY_LESION: ["lesion", "abnormal tissue"],
    Topic.OPACITY_NODULES: ["nodule", "small mass"],
    Topic.OPACITY_MASS: ["mass", "large mass", "tumor"],
    Topic.OPACITY_CYST: ["cyst", "bulla", "bleb"],
    Topic.LUNG_HYPERINFLATION: ["lung hyperinflation", "overinflated lungs"],
    Topic.PLEURAL_ABNORMALITY: ["pleural abnormality", "pleural issue"],
    Topic.PLEURAL_PNEUMOTHORAX: ["pneumothorax", "collapsed lung"],
    Topic.PLEURAL_EFFUSION: ["pleural effusion", "fluid in pleural space"],
    Topic.PLEURAL_THICKENING: ["pleural thickening", "thickened pleura"],
    Topic.MEDIASTINAL_ENLARGEMENT: ["mediastinal enlargement", "enlarged mediastinum"],
    Topic.MEDIASTINAL_ECTATIC_AORTA: ["ectatic aorta", "dilated aorta"],
    Topic.MEDIASTINAL_CALCIFIED_AORTA: ["calcified aorta", "aortic calcification"],
    Topic.CARDIAC_CARDIOMEGALY: ["cardiomegaly", "enlarged heart"],
    Topic.CARDIAC_PNEUMOPERICARDIUM: ["pneumopericardium", "air in pericardium"],
    Topic.CARDIAC_PERICARDIAL_EFFUSION: ["pericardial effusion", "fluid around heart"],
    Topic.HILAR_ENLARGEMENT: ["hilar enlargement", "enlarged hilum"],
    Topic.CHEST_WALL_SUBCUTANEOUS: ["subcutaneous emphysema", "air under skin"],
    Topic.BONE_RIB_FRACTURE: ["rib fracture", "broken rib"],
    Topic.BONE_SPINE_DEGENERATIVE: ["spine degeneration", "degenerative changes in spine"],
    Topic.BONE_SPINE_SCOLIOSIS: ["scoliosis", "curved spine"],
    Topic.BONE_SPINE_COMPRESSION_FRACTURE: ["compression fracture", "spine compression fracture"],
    Topic.BONE_SHOULDER_DEGENERATIVE: ["shoulder degeneration", "degenerative changes in shoulder"],
    Topic.BONE_SHOULDER_DISLOCATION: ["shoulder dislocation", "dislocated shoulder"],
    Topic.BONE_SHOULDER_FRACTURE: ["shoulder fracture", "broken shoulder"],
    Topic.FOREIGN_BUTTON: ["button"],
    Topic.FOREIGN_HAIR: ["hair"],
    Topic.FOREIGN_GRAVEL: ["gravel"],
    Topic.FOREIGN_NEEDLE: ["needle"],
    Topic.FOREIGN_BULLET: ["bullet"],
    Topic.DEVICE_TUBE: ["tube"],
    Topic.DEVICE_FEEDING_TUBE: ["feeding tube"],
    Topic.DEVICE_CHEST_TUBE: ["chest tube"],
    Topic.DEVICE_PACEMAKER: ["pacemaker"],
    Topic.DEVICE_ICD: ["ICD", "implantable cardioverter-defibrillator"],
    Topic.DEVICE_LVAD: ["LVAD", "left ventricular assist device"],
    Topic.DEVICE_ORTHOPEDIC_WIRE: ["orthopedic wire"],
    Topic.DEVICE_ORTHOPEDIC_PLATE: ["orthopedic plate"],
    Topic.DEVICE_ORTHOPEDIC_SCREW: ["orthopedic screw"],
    Topic.DEVICE_ORTHOPEDIC_NAIL: ["orthopedic nail"],
    Topic.DEVICE_SURGICAL_STAPLES: ["surgical staples"],
    # Topic.LUNG_VOLUMES: ["lung volumes", "lung capacity"],
    # Topic.LUNG_PNEUMONIA: ["pneumonia", "lung infection"],
    # Topic.LUNG_PULMONARY_EDEMA: ["pulmonary edema", "fluid in lungs"],
    # Topic.LUNG_SCARRING: ["lung scarring", "pulmonary fibrosis"],
    # Topic.LUNG_CLEARNESS: ["lung clearness", "clear lungs"],
    # Topic.HEART_SIZE: ["heart size", "cardiac size"],
    # Topic.HEART_CONTOUR: ["heart contour", "cardiac contour"],
    # Topic.HEART_PULMONARY_VASCULAR_CONGESTION: ["pulmonary vascular congestion", "lung congestion"],
    # Topic.HEART_HYPERTENSION: ["heart hypertension", "high blood pressure"],
    # Topic.BONE_TISSUE: ["bone tissue", "osseous tissue"],
    # Topic.BONE_DENSITY: ["bone density", "bone mass"],
    # Topic.HEMIDIAPHRAGM: ["hemidiaphragm", "half of diaphragm"]
}

def estimate_cost(prompt_tokens, completion_tokens):
    input_cost = 0.03
    output_cost = 0.06
    
    return (input_cost*prompt_tokens/1000 + output_cost*completion_tokens/1000)

def query_openai(instructions, report, max_tokens=500, temperature=.4):
    API_VERSION = "2023-05-15"
    GPT_MODEL = 'gpt41106'

    client = AzureOpenAI(
        api_key = API_KEY, 
        api_version = API_VERSION,
        azure_endpoint="https://xzhang.openai.azure.com"
    )

    setup = """
        Suppose you are an expert radiologist and are given a radiology report writen by your assistant. 
        Give specific instructions to your assistant on modifying the report. 
        I will provide you with the type of instructions to make and the clinical topics to focus on.  
        There is no imaging involved. Create instructions that are specific, well-defined, concise, and applicable to the report. 
        If the instruction I asked you to make does not apply to the report, you can create one that does. 
        If there are multiple instructions for a report, they should be based on the original report.
        If an instruction applies to multiple instances in the report, make sure to make all adjustments accordingly in the modified report.
        Renumber the modified report at the end if needed. For your reply, format them this way: 
        Instructions: Instruction 1:.... Insturction 2:.... \n Modified Report: ...
        Follow this example:
    """

    example1 = """
        Example 1: Create the following instruction(s): an instruction to adds an observation to the entire report about consolidation; 
        an instruction to remove an observation about cardiac silhouette in the impression section;
        Original report:  
        FINDINGS:
        1. AP and lateral views of the chest were obtained.
        2. The right costophrenic angle is not fully included on the image.
        IMPRESSION: 
        3. Top normal cardiac silhouette without pleural effusion or pulmonary edema.
        Instructions: Instruction 1: Add no focal consolidation, pleural effusion, pneumothorax to both the findings and impression sections. Instruction 2: Remove Line 3 in the original report.  
        Modified Report:  
        FINDINGS:
        1. AP and lateral views of the chest were obtained.
        2. The right costophrenic angle is not fully included on the image.
        3. No focal consolidation, pleural effusion, or evidence of pneumothorax is seen.
        IMPRESSION: 
        4. No focal consolidation, pleural effusion, or evidence of pneumothorax 
    """
    
    example2 = """
        Example 2: Create the following instruction(s): an instruction to change the anatomical location of an observation.
        Then provide the modified report.
        Original report: 
        FINDINGS: 
        1. Endotracheal tube is seen with tip in the right mainstem bronchus.
        2. Hazy right basilar opacity may be due to atelectasis. 
        3. Right mainstem intubation is seen.
        IMPRESSION: 
        4. Right mainstem intubation.
        Instruction 1: Change intubation to left. 
        Modified report:
        FINDINGS: 
        1. Endotracheal tube is seen with tip in the right mainstem bronchus.
        2. Hazy right basilar opacity may be due to atelectasis.
        3. Left mainstem intubation is seen.
        IMPRESSION: 
        3. Left mainstem intubation.
    """


    # print(setup+example1+example2)
    try:
        response = client.chat.completions.create(
            model=GPT_MODEL, 
            messages = [
                {'role': 'system', 'content': setup+example1+example2},
                # {'role': 'system', 'content': setup+example1},
                {'role': 'user', 'content': 
                    f"For the following original report, {instructions}. Original report:\n {report}"}
            ], 
            max_tokens=max_tokens,
            temperature=temperature
        )
        completion=response.choices[0].message.content
        cost = estimate_cost(response.usage.prompt_tokens, response.usage.completion_tokens)
        return completion, cost
    
    except Exception as e:
        print(e)
        # time.sleep(5)
        return query_openai(instructions, report, max_tokens, temperature)


class GeneratedDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    ids = [item['id'] for item in batch]
    instructions = [item['instructions'] for item in batch]
    originals = [item['report_text'] for item in batch]
    gts = [item['modified_text'] for item in batch]
    return ids, instructions, originals, gts
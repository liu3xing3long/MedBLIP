import io
import os
import random

import pyarrow as pa
import torch
from PIL import Image

from .utils import record_ent_ref, create_pos_matrix
from .transforms import keys_to_transforms
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling, BertTokenizerFast, RobertaTokenizerFast
from .data_collator import DataCollatorForWholeEntityMask

FG_TEXT_LIST = ['normal', 'pleural effusion', 'opacity', 'pneumothorax', 'edema', 'atelectasis',  'tube', 'consolidation','enlarged cardiomediastinum','tip', 'pneumonia','line','cardiomegaly', 'fracture','calcification',
                'device','engorgement',  'nodule', 'wire',  'pacemaker', 'pleural thicken', 'marking', 'scar', 'hyperinflate', 'blunt',  'collapse', 'emphysema', 'aerate', 'mass','infiltration', 'obscure', 'deformity', 'hernia',
                'drainage', 'distention', 'shift', 'stent', 'lesion', 'hardware', 'dilation',  'aspiration']

def get_pretrained_tokenizer(from_pretrained):
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            if 'roberta' in from_pretrained:
                RobertaTokenizerFast.from_pretrained(from_pretrained)
            elif 'bert' in from_pretrained.lower():
                BertTokenizerFast.from_pretrained(from_pretrained, do_lower_case="uncased" in from_pretrained)
        torch.distributed.barrier()

    tokenizer = None
    if 'roberta' in from_pretrained:
        tokenizer = RobertaTokenizerFast.from_pretrained(from_pretrained)
    elif 'bert' in from_pretrained.lower():
        tokenizer = BertTokenizerFast.from_pretrained(from_pretrained, do_lower_case="uncased" in from_pretrained)
    return tokenizer


class BaseDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            data_dir: str,
            transform_keys: list,
            image_size: int,
            names: list,
            text_column_name: str = "",
            max_text_len: int = 40,
            draw_false_image: int = 0,
            draw_false_text: int = 0,
            image_only: bool = False,
            label_column_name: str = "",
            max_num_ents: int = 24,
    ):
        super().__init__()
        assert len(transform_keys) >= 1
        # Hyper-Parameters
        self.text_column_name = text_column_name
        self.names = names
        self.max_text_len = max_text_len
        self.max_num_ents = max_num_ents
        self.draw_false_image = draw_false_image
        self.draw_false_text = draw_false_text
        self.image_only = image_only
        self.data_dir = data_dir
        self.label_column_name = label_column_name

        # Image Transformations
        if "train" not in names[0]:
            transform_keys = [transform_key.replace("_randaug", "") for transform_key in transform_keys]
            transform_keys = [transform_key.replace("_resizedcrop", "") for transform_key in transform_keys]
        self.transforms = keys_to_transforms(transform_keys, size=image_size)
        self.clip_transform = False
        for transform_key in transform_keys:
            if 'clip' in transform_key:
                self.clip_transform = True
                break
        
        # Read Texts
        if len(names) != 0:
            tables = [
                pa.ipc.RecordBatchFileReader(pa.memory_map(f"{data_dir}/pretrain_arrows_umls/{name}.arrow", "r")).read_all()
                for name in names
                if os.path.isfile(f"{data_dir}/pretrain_arrows_umls/{name}.arrow")
            ]
            self.table_names = list()
            for i, name in enumerate(names):
                self.table_names += [name] * len(tables[i])
            self.table = pa.concat_tables(tables, promote=True)
            if text_column_name != "":
                self.text_column_name = text_column_name
                self.all_texts = self.table[text_column_name].to_pandas().tolist()
                assert type(self.all_texts[0][0]) == str
            else:
                self.all_texts = list()
        else:
            self.all_texts = list()

        # Read Entities
        self.all_img_ents = self.table["img_ents"].to_pandas().tolist()
        self.all_txt_ents = self.table["txt_ents"].to_pandas().tolist()
        print('all_img_ents length: ', len(self.all_img_ents))
        print('all_txt_ents length: ', len(self.all_txt_ents))

        ########################################################################
        # read labels
        fulllabels = open(fr"{data_dir}/fg_radgraph_metric.csv").read().strip().split("\n")[1:]
        imageid2labels = {os.path.basename(fl.split(',')[0]): fl.split(',')[1: ] for fl in fulllabels}
        # link labels
        self.all_fg_radgraph_labels = []
        fullimageids = self.table["image_id"].to_pandas().tolist()
        valid_all_fg_radgraph_labels = 0
        for fiid in fullimageids:
            iid = os.path.basename(fiid)
            str_labels = []
            # str_labels = [os.path.splitext(iid)[0], ]
            if iid in imageid2labels:
                this_label_list = imageid2labels[iid]
                # print(fr'{iid}  -> {this_label_list}')
                for idx, tl in enumerate(this_label_list):
                    if eval(tl) >= 1:
                        str_labels.append(FG_TEXT_LIST[idx])
                    else:
                        pass
            else:
                pass
            if len(str_labels) > 0:
                # print(fr'imageid {iid}, labels {str_labels}')
                valid_all_fg_radgraph_labels += 1
            self.all_fg_radgraph_labels.append(str_labels)
        print('all_fg_radgraph_labels length: ', len(self.all_fg_radgraph_labels), 
              'validlength: ', valid_all_fg_radgraph_labels)
        print('image length: ', len(self.table['image']))

        ########################################################################
        self.ent2id = open(fr"{data_dir}/knowledge/entity2id.txt").read().strip().split("\n")[1:]
        self.ent2id = {kv.split("\t")[0]: kv.split("\t")[2] for kv in self.ent2id}
        self.id2ent = {v: k for k, v in self.ent2id.items()}

        # Record Index Mappings
        self.index_mapper = dict()
        if text_column_name != "" and not self.image_only:
            j = 0
            for i, texts in enumerate(self.all_texts):
                for _j in range(len(texts)):
                    self.index_mapper[j] = (i, _j)
                    j += 1
        else:
            for i in range(len(self.table)):
                self.index_mapper[i] = (i, None)
        print('index_mapper length: ', len(self.index_mapper))

        ###########################################################################################
        # Tokenizer
        tokenizer = 'bert-base-uncased'
        whole_word_masking = True
        mlm_prob = 0.15
        
        self.tokenizer = get_pretrained_tokenizer(tokenizer)
        self.vocab_size = self.tokenizer.vocab_size

        # Collator for Dataloaders
        collator = (
            DataCollatorForWholeEntityMask
            if whole_word_masking
            else DataCollatorForLanguageModeling
        )

        self.tokenizer = get_pretrained_tokenizer(tokenizer)
        self.mlm_collator = collator(tokenizer=self.tokenizer, mlm=True, mlm_probability=mlm_prob)
        ###########################################################################################

    @property
    def corpus(self):
        return [text for texts in self.all_texts for text in texts]

    def __len__(self):
        return len(self.index_mapper)

    def get_strlabels(self, index):
        index, caption_index = self.index_mapper[index]
        return self.all_fg_radgraph_labels[index]

    def get_raw_image(self, index, image_key="image"):
        index, caption_index = self.index_mapper[index]
        image_bytes = io.BytesIO(self.table[image_key][index].as_py())
        image_bytes.seek(0)
        if self.clip_transform:
            return Image.open(image_bytes).convert("RGBA")
        else:
            return Image.open(image_bytes).convert("RGB")

    def get_image(self, index, image_key="image"):
        image = self.get_raw_image(index, image_key=image_key)
        image_tensor = [tr(image) for tr in self.transforms]
        return {
            "image": image_tensor,
            "img_index": self.index_mapper[index][0],
            "cap_index": self.index_mapper[index][1],
            "raw_index": index,
        }

    def get_false_image(self, rep, image_key="image", selected_index=None):
        random_index = random.randint(0, len(self.index_mapper) - 1)
        image = self.get_raw_image(random_index, image_key=image_key)
        image_tensor = [tr(image) for tr in self.transforms]
        return {f"false_image_{rep}": image_tensor}

    def get_text(self, raw_index):
        index, caption_index = self.index_mapper[raw_index]
        text = self.all_texts[index][caption_index]

        #############################################
        radgraph_text_list = self.get_strlabels(raw_index)
        if len(radgraph_text_list) > 0:
            text += fr'. The diagnosis is {" ".join(radgraph_text_list)}.'
        #############################################

        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
            return_special_tokens_mask=True,
            return_offsets_mapping=True,
        )
        encoding = record_ent_ref(encoding, self.all_txt_ents[index][caption_index])
        img_label = torch.zeros(len(self.ent2id), dtype=torch.long)
        txt_label = torch.zeros(len(self.ent2id), dtype=torch.long)
        for label_index in self.all_img_ents[index]:
            img_label[label_index] = 1
        for label_index in encoding["txt_label"]:
            txt_label[label_index] = 1
        return {
            "text": (text, encoding),
            "img_index": index,
            "cap_index": caption_index,
            "raw_index": raw_index,
            "img_label": img_label,
            "txt_label": txt_label,
        }

    def get_false_text(self, rep, selected_index=None):
        random_index = random.randint(0, len(self.index_mapper) - 1)
        index, caption_index = self.index_mapper[random_index]
        text = self.all_texts[index][caption_index]
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
            return_special_tokens_mask=True,
            return_offsets_mapping=True,
        )
        encoding = record_ent_ref(encoding, self.all_txt_ents[index][caption_index])
        return {f"false_text_{rep}": (text, encoding)}

    def get_suite(self, index):
        result = None
        while result is None:
            try:
                ret = dict()
                ret.update(self.get_image(index))
                if not self.image_only:
                    txt = self.get_text(index)
                    ret.update({"replica": True if txt["cap_index"] > 0 else False})
                    ret.update(txt)
                for i in range(self.draw_false_image):
                    ret.update(self.get_false_image(i, selected_index=index))
                for i in range(self.draw_false_text):
                    ret.update(self.get_false_text(i, selected_index=index))
                result = True
            except Exception as e:
                print(f"Error while read file idx {index} in {self.names[0]} -> {e}")
                index = random.randint(0, len(self.index_mapper) - 1)
        return ret

    def collate(self, batch, mlm_collator=None):
        if mlm_collator is None:
            mlm_collator = self.mlm_collator

        batch_size = len(batch)
        keys = set([key for b in batch for key in b.keys()])
        dict_batch = {k: [dic[k] if k in dic else None for dic in batch] for k in keys}

        img_keys = [k for k in list(dict_batch.keys()) if "image" in k]
        img_sizes = list()

        for img_key in img_keys:
            img = dict_batch[img_key]
            img_sizes += [ii.shape for i in img if i is not None for ii in i]

        for size in img_sizes:
            assert (len(size) == 3), f"Collate error, an image should be in shape of (3, H, W), instead of given {size}"

        if len(img_keys) != 0:
            max_height = max([i[1] for i in img_sizes])
            max_width = max([i[2] for i in img_sizes])

        #####################################################################
        IMAGE_DEPTH = 1
        for img_key in img_keys:
            img = dict_batch[img_key]
            view_size = len(img[0])
            # new_images = [torch.zeros(batch_size, 3, max_height, max_width) for _ in range(view_size)]
            new_images = [torch.zeros(batch_size, IMAGE_DEPTH, 3, max_height, max_width) for _ in range(view_size)]
            for bi in range(batch_size):
                orig_batch = img[bi]
                for vi in range(view_size):
                    if orig_batch is None:
                        new_images[vi][bi] = None
                    else:
                        orig = img[bi][vi]
                        # new_images[vi][bi, :, : orig.shape[1], : orig.shape[2]] = orig
                        # duplicate image
                        for ii in range(IMAGE_DEPTH):
                            new_images[vi][bi, ii, :, : orig.shape[1], : orig.shape[2]] = orig
            dict_batch[img_key] = new_images
        #####################################################################

        if "text" in dict_batch:
            dict_batch["img_labels"] = torch.vstack([d["img_label"] for d in batch])
            dict_batch["txt_labels"] = torch.vstack([d["txt_label"] for d in batch])

        txt_keys = [k for k in list(dict_batch.keys()) if "text" in k]
        if len(txt_keys) != 0:
            encodings = [[d[1] for d in dict_batch[txt_key]] for txt_key in txt_keys]
            flatten_encodings = [e for encoding in encodings for e in encoding]
            flatten_mlms = mlm_collator(flatten_encodings)

            ent_inputs, ent_inputs_mlm = [], []
            for e, m in zip(flatten_encodings, flatten_mlms["labels"]):
                ent_inputs.append(create_pos_matrix(e, self.max_text_len, self.max_num_ents))
                ent_inputs_mlm.append(create_pos_matrix(e, self.max_text_len, self.max_num_ents, m))
            flatten_pos_matrices = torch.stack([item[0] for item in ent_inputs])
            flatten_ent_ids = torch.stack([item[1] for item in ent_inputs])
            flatten_ent_masks = torch.stack([item[2] for item in ent_inputs])
            flatten_pos_matrices_mlm = torch.stack([item[0] for item in ent_inputs_mlm])
            flatten_ent_ids_mlm = torch.stack([item[1] for item in ent_inputs_mlm])
            flatten_ent_masks_mlm = torch.stack([item[2] for item in ent_inputs_mlm])
            for i, txt_key in enumerate(txt_keys):
                texts, encodings = ([d[0] for d in dict_batch[txt_key]], [d[1] for d in dict_batch[txt_key]])
                mlm_ids, mlm_labels = (
                    flatten_mlms["input_ids"][batch_size * (i): batch_size * (i + 1)],
                    flatten_mlms["labels"][batch_size * (i): batch_size * (i + 1)],
                )
                pos_matrices, ent_ids, ent_masks = (
                    flatten_pos_matrices[batch_size * (i): batch_size * (i + 1)],
                    flatten_ent_ids[batch_size * (i): batch_size * (i + 1)],
                    flatten_ent_masks[batch_size * (i): batch_size * (i + 1)],
                )
                pos_matrices_mlm, ent_ids_mlm, ent_masks_mlm = (
                    flatten_pos_matrices_mlm[batch_size * (i): batch_size * (i + 1)],
                    flatten_ent_ids_mlm[batch_size * (i): batch_size * (i + 1)],
                    flatten_ent_masks_mlm[batch_size * (i): batch_size * (i + 1)],
                )

                input_ids = torch.zeros_like(mlm_ids)
                attention_mask = torch.zeros_like(mlm_ids)
                for _i, encoding in enumerate(encodings):
                    _input_ids, _attention_mask = (
                        torch.tensor(encoding["input_ids"]),
                        torch.tensor(encoding["attention_mask"]),
                    )
                    input_ids[_i, : len(_input_ids)] = _input_ids
                    attention_mask[_i, : len(_attention_mask)] = _attention_mask

                dict_batch[txt_key] = texts
                dict_batch[f"{txt_key}_ids"] = input_ids
                dict_batch[f"{txt_key}_labels"] = torch.full_like(input_ids, -100)
                dict_batch[f"{txt_key}_ids_mlm"] = mlm_ids
                dict_batch[f"{txt_key}_labels_mlm"] = mlm_labels
                dict_batch[f"{txt_key}_masks"] = attention_mask

                dict_batch[f"{txt_key}_pos_matrices"] = pos_matrices
                dict_batch[f"{txt_key}_ent_ids"] = ent_ids
                dict_batch[f"{txt_key}_ent_masks"] = ent_masks
                dict_batch[f"{txt_key}_pos_matrices_mlm"] = pos_matrices_mlm
                dict_batch[f"{txt_key}_ent_ids_mlm"] = ent_ids_mlm
                dict_batch[f"{txt_key}_ent_masks_mlm"] = ent_masks_mlm

        return dict_batch

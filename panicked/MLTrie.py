# 本库所使用的marisa_trie库以BSD 2-clause协议发布。所使用的numpy库以NumPy License发布。所使用的tqdm库以MPL-2.0 & MIT协议发布。所使用的Python标准库以PSF License发布。
# 本库的其余所有部分以WTFPL协议发布。

import gc
import pickle
import random
import struct
from collections import Counter, defaultdict, deque
from collections.abc import Generator, Iterable, Iterator, Sequence
from contextlib import suppress
from itertools import chain
from typing import Any, cast

import marisa_trie
import numpy as np
import tqdm
from numpy.typing import NDArray

bit = uint7 = uint8 = uint24 = uint32 = int
byte3 = byte4 = bytes

type Tnode = tuple[int, uint24]
type Tnodes = tuple[int, uint24, uint8]
type TnodeData = tuple[uint24, uint8, uint24, uint8, uint24, bit, uint7]

NS = 12  # Node Size
POE = 3  # Parent index Offset End
PSO = 8  # Parent index Shift Offset
HOF = 3  # cHar OFfset
HOE = 4  # cHar Offset End
HMS = 0x00_00_00_FF  # cHar MaSk
COF = 4  # Childern index OFfset
COE = 7  # Childern index Offset End
CSO = 8  # Childern index Shift Offset
LOF = 7  # childern Length OFfset
LOE = 8  # childern Length Offset End
LMS = 0x00_00_00_FF  # childern Length MaSk
VOF = 8  # Value OFfset
VOE = 11  # Value Offset End
VSO = 8  # Value Shift Offset
JOF = 11  # Jump OFfset
JSO = 7  # Jump Shift Offset
JMS = 0b1  # Jump MaSk （注: 此掩码是先右移JSO位之后再应用的）
MOF = 11  # Max length OFfset
MMS = 0x7F  # Max length MaSk
PDFT = 0xFF_FF_FF  # Parent DeFaulT value
CDFT = 0xFF_FF_FF  # Childern index DeFaulT value
LDFT = 0x00  # childern Length DeFaulT value
VDFT = 0xFF_FF_FF  # Value DeFaulT value
JDFT = 0b1  # Jump DeFaulT value
MLLEN = 0xFF_FF_FD  # Max Layer LENgth
MVCNT = 0xFF_FF_FD  # Max Value CouNT
MDPTH = 0x7F  # Max DePTH

JSFT = JDFT << JSO  # Jump ShiFTed

SR = 16  # Sample Rate，建议取到8~16以保证均匀性，但是大了会慢
SSVR = 0.05  # Space Separated Value Ratio，SSV占比。用于extract_unique，过大会浪费时间，过小会增加重试次数也浪费时间

# 关于nidx的命名规范：nidx（Node InDeX）表示索引位置，nmidx（Node Memory InDeX）表示内存位置，nidx = nmidx * NS。
# 其余idx/midx同理（不包括lidx、vidx）。


# @jit(nopython=True, nogil=True, cache=True)
def headfindHOE(arr: bytearray | memoryview, target: bytes | bytearray):
    lo = 0
    hi = len(arr) // NS
    while lo < hi:
        mid = (lo + hi) // 2
        if arr[mid * NS : mid * NS + HOE] < bytearray(target):
            lo = mid + 1
        else:
            hi = mid

    if lo * NS + NS > len(arr):
        return -1
    if arr[lo * NS : lo * NS + HOE] == target:
        return lo
    return -1


class SourceDrainedError(OverflowError):
    pass


class SourceEmptyError(OverflowError):
    pass


class MLTrie:

    def __init__(self, data: dict[bytes, str] | None = None):

        # 能处理的最大数据量是每层16777214个节点、16777214个值，虽然说多数情况下理应是够用的，但我还是非常不建议拿它存几GB的东西
        # 我的使用场景就是github: WFLing-seaer/HongDict的那些文件，加上拼音之后大概0.98GB，已经非常逼近每层节点数的极限了
        # 再继续加数据量的话就得改数据结构了
        # 另外这个数据结构几乎没有压缩，就算dump到.bin了也是没压缩的。用LZMA2压缩一下的话压缩率能有（非常惊人的）14%（就我的数据而言）
        # 但是我dump的时候默认不压缩是因为压缩有点太吃性能了，而且不压缩的话加载几GB的.bin也只是毫秒~亚秒级的，压缩的话就不知道得加载多久了
        # 总而言之就是个自己用的东西，也没指望着能适配所有人的工况，反正是开源的东西自己想咋改咋改吧。甚至WTFPL协议（目移）
        # 已无力，已摆烂（蠕动离开

        # TODO: 将节点分为AB类，A类不存储val jump mlen。对于双拼（声+韵+调）来说排布就是AAB AAB这样的，大概能省20%空间，但是性能肯定会下降一点

        # 以及因为这里面涉及太多神必常量&切片了，所以注释我尽可能写的完整一些

        main_progress = tqdm.tqdm(
            desc="排序中…… (1/6)",
            total=5,
            leave=None,  # type: ignore # 注: tqdm手册中写的，传入None以仅保留首行进度条。它自己类型注解里倒是没写|None，怒哩
            unit_scale=False,
            bar_format="{desc} | {bar} |",
            position=0,
        )

        data = data or {}
        sorted_data = sorted(data.items(), key=lambda x: x[0])
        # 排序的作用是，保证键是字典序的，这样插入的时候就能保证永远是插入在层尾，可以避免插入时移动后续位的开销

        main_progress.update(1)
        main_progress.set_description("分组中…… (2/6)")

        lengths: defaultdict[int, list[str]] = defaultdict(list[str])
        for key, vals in tqdm.tqdm(data.items(), position=1, leave=None, desc="--"):  # type: ignore # 理由同上
            lengths[len(key) - 1].append(vals)

        del data  # 因为数据量太大了，删一下能少用点内存，要不然构建过程中将近10GB的内存占用很多设备顶不住的
        gc.collect()

        main_progress.update(1)
        main_progress.set_description("写MARISA中…… (3/6)")

        self.lval: dict[int, marisa_trie.Trie] = {
            k: marisa_trie.Trie(v)
            for k, v in tqdm.tqdm(
                lengths.items(),
                position=1,
                leave=None,  # type: ignore # 理由同上
                desc="--",
            )
        }
        if any(len(lv) > MVCNT for lv in self.lval.values()):
            raise OverflowError

        main_progress.update(1)
        main_progress.set_description("构建layers中…… (4/6)")

        layers: list[bytearray] = []

        # R: 构建layers 开始
        for key, val in tqdm.tqdm(sorted_data, position=1, leave=None, desc="--"):  # type: ignore # 理由同上
            key = cast(bytes, key)  # 为什么tqdm返回的值类型全变成Unknown了。怒哩×2
            val = cast(str, val)
            if not key:
                continue
            parent: int = PDFT

            # R: 逐位处理单个键 开始
            for lidx, char in enumerate(key):
                head: byte4 = ((parent << PSO) | char).to_bytes(4, "big")
                depth_left: uint7 = min(len(key) - lidx - 1, MDPTH)  # 0x7F表示所有最大深度在128及以上的节点。因为一共也没几个（）

                if lidx >= len(layers):
                    layer = bytearray()
                    layers.append(layer)
                    nidx = -1  # 插入新层了就不用查找有没有了，因为肯定是没有（
                else:
                    layer = layers[lidx]
                    nidx = headfindHOE(layer, head)

                if nidx == -1:
                    # 本层不存在该字符，插入一下
                    nidx = len(layer) // NS
                    # 直接用len是因为，一开始的排序保证了新增的节点必定在层尾
                    # （第一层有序，则第二层前3B有序，因为第二层char有序，则第二层有序，归纳法可得全部有序）
                    layer += struct.pack(">3I", (parent << 8) | char, (CDFT << 8) | LDFT, (VDFT << 8) | JSFT | depth_left)
                    # jump位默认为1（无值）
                    if lidx:  # lidx为0时是第一层，父节点为根节点；根节点不存在因而不用更新。
                        # 更新父节点的子节点信息
                        pmidx = parent * NS
                        if not layers[lidx - 1][pmidx + LOF]:  # 子节点计数为0（还没有子节点）
                            layers[lidx - 1][pmidx + COF : pmidx + COE] = nidx.to_bytes(3, "big")  # 写入子节点起始索引
                        layers[lidx - 1][pmidx + LOF] += 1  # parent * NS + LOF是子节点数量字段
                else:
                    layer[nidx * NS + MOF] = (max(layer[nidx * NS + MOF] & MMS, depth_left)) | (layer[nidx * NS + MOF] & JSFT)  # 更新最大深度
                parent = nidx
            # R: 逐位处理单个键 结束

            # R: 处理键尾 开始
            vidx = self.lval[len(key) - 1][val]  # 不用try（必有）
            layers[len(key) - 1][parent * NS + VOF : parent * NS + VOE] = vidx.to_bytes(3, "big")
            layers[len(key) - 1][parent * NS + JOF] &= MMS  # 将jump位置0（&MMS，仅保留MDPT字段）表示有值
            # R: 处理键尾 结束

        # R: 构建layers 结束

        main_progress.update(1)
        main_progress.set_description("写入跳转信息中…… (5/6)")

        # R: 写入跳转信息 开始
        for layer in tqdm.tqdm(layers, position=1, desc="--", leave=None):  # type: ignore # 理由同上
            last_valued_nidx_bytes = VDFT.to_bytes(3, "big")
            for nmidx in range(len(layer) - NS, -1, -NS):
                # 倒序遍历。因为跳转信息指向的是下一个有值节点
                if layer[nmidx + JOF] & JSFT:  # JUMP位为1，表示无值
                    layer[nmidx + VOF : nmidx + VOE] = last_valued_nidx_bytes
                else:
                    last_valued_nidx_bytes = (nmidx // NS).to_bytes(3, "big")
        # R: 写入跳转信息 结束

        main_progress.update(1)
        main_progress.set_description("收尾中…… (6/6)")

        self.layers: list[bytes] = list(map(bytes, layers))

        del layers
        gc.collect()

        self.check_integrity()

        main_progress.update(1)
        main_progress.set_description("完成")
        main_progress.close()

    def dumps(self):
        components: bytearray = bytearray(b"MLTW")  # 文件头
        marisa_data = pickle.dumps(self.lval)

        # 元数据: marisa数据长度以及各层长度
        components += len(marisa_data).to_bytes(4, "big")
        components += len(self.layers).to_bytes(4, "big")
        for layer in self.layers:
            components += len(layer).to_bytes(4, "big")
        components += marisa_data

        current_length = len(components)
        padding_needed = (NS - (current_length % NS)) % NS
        components += b"\x00" * padding_needed
        # 别问我为啥要对齐到NS字节。真的只是为了好看，真的，除此以外没有任何作用，你愿意的话删了也无所谓
        # 当然删了的话load也得改一下

        components += b"".join(self.layers)
        return bytes(components)

    def dump(self, fp: str):
        with open(fp, "wb") as f:
            f.write(self.dumps())

    @classmethod
    def loads(cls, data: bytes):
        if data[:4] != b"MLTW":
            raise ValueError

        marisa_len, num_layers = struct.unpack_from(">2I", data, 4)
        offset = 12

        if num_layers > 0:
            layer_lengths = list(struct.unpack_from(f">{num_layers}I", data, 12))
            offset += 4 * num_layers
        else:
            layer_lengths = []

        marisa_data = data[offset : offset + marisa_len]
        offset += marisa_len

        padding_needed = (NS - (offset % NS)) % NS
        offset += padding_needed

        layers = []
        for layer_len in layer_lengths:
            layer_data = data[offset : offset + layer_len]
            layers.append(layer_data)
            offset += layer_len

        instance = cls.__new__(cls)
        instance.lval = pickle.loads(marisa_data)
        instance.layers = layers
        instance.check_integrity()
        return instance

    @classmethod
    def load(cls, fp: str):
        instance = cls.__new__(cls)
        instance.layers = []

        with open(fp, "rb") as f:
            magic = f.read(4)
            if magic != b"MLTW":
                raise ValueError

            header_data = f.read(8)
            marisa_len, num_layers = struct.unpack(">2I", header_data)

            layer_lengths = []
            if num_layers > 0:
                layer_lengths_data = f.read(4 * num_layers)
                layer_lengths = list(struct.unpack(f">{num_layers}I", layer_lengths_data))

            marisa_data = f.read(marisa_len)

            current_pos = f.tell()
            padding_needed = (NS - (current_pos % NS)) % NS
            if padding_needed > 0:
                f.read(padding_needed)

            for layer_len in layer_lengths:
                instance.layers.append(f.read(layer_len))

            instance.lval = pickle.loads(marisa_data)

        instance.check_integrity()

        return instance

    @staticmethod
    def _unpack(data: bytes) -> TnodeData:
        pidx_char, cidx_clen, val_jmp_mdpt = struct.unpack(">3I", data)
        return (
            uint24(pidx_char >> PSO),
            uint8(pidx_char & HMS),
            uint24(cidx_clen >> CSO),
            uint8(cidx_clen & LMS),
            uint24(val_jmp_mdpt >> VSO),
            bit(val_jmp_mdpt >> JSO & JMS),
            uint7(val_jmp_mdpt & MMS),
        )

    @staticmethod
    def seg_merge(segs: Iterable[Tnode | Tnodes]) -> list[Tnodes]:
        src: list[Tnodes] = [nodes if len(nodes) == 3 else (*nodes, 1) for nodes in segs]
        if not src:
            return []
        src.sort()  # sort默认是把元组按照首位优先排序
        ret: list[Tnodes] = [src.pop(0)]
        for nodes in src:
            if nodes[0] == ret[-1][0] and nodes[1] == ret[-1][1] + ret[-1][2]:
                ret[-1] = (nodes[0], ret[-1][1], ret[-1][2] + nodes[2])
            else:
                ret.append(nodes)
        return ret

    @staticmethod
    def expand(nodes: Tnodes) -> Generator[Tnode, None, None]:
        for nidx_offset in range(nodes[2]):
            yield nodes[0], nodes[1] + nidx_offset

    @staticmethod
    def expand_batch(segs: Sequence[Tnodes]) -> Generator[Tnode, None, None]:
        for nodes in MLTrie.seg_merge(segs):
            yield from MLTrie.expand(nodes)

    def expand_leaf(self, nodes: Tnodes) -> Generator[Tnode, None, None]:
        layer = memoryview(self.layers[nodes[0]])
        nidx = nodes[1]
        nidx_end = nodes[1] + nodes[2]
        while nidx < nidx_end:
            if layer[nidx * NS + JOF] & JSFT:  # jump位为1
                nidx = struct.unpack_from(">I", layer, nidx * NS + VOF)[0] >> VSO
            else:
                yield nodes[0], nidx
                nidx += 1

    def expand_leaf_batch(self, segs: Sequence[Tnodes]) -> Generator[Tnode, None, None]:
        for nodes in self.seg_merge(segs):
            yield from self.expand_leaf(nodes)

    def extract(self, nodes: Tnodes, num: int, exact: bool = False) -> list[str]:
        # 逆概率校正权重：1 / (Dmax * (1 + 1 / C))，Dmax是每段中采样点到段尾的最大距离，C是段内采样点数
        # 在采样点数约等于或者略小于段数的时候能有（还算）不错的效果，在段分布不是非常不均匀的情况下概率差在30%以下
        # 算法如下: 目标区域内均匀随机取点，计算点到段尾距离&每段被选中频率，然后用上式算IPW
        try:
            marisa = self.lval[nodes[0]]
        except IndexError:
            if exact:
                raise
            return []

        trial = min(num * SR, nodes[2])  # 用16倍来获得一个还算平均的概率。倍数越大概率越平均，当然也越慢
        while True:
            samples: NDArray[np.long] = np.random.choice(nodes[2], trial, replace=True)
            samples += nodes[1]
            samples.sort()
            tails: NDArray[np.uint32] = np.array(
                [
                    (
                        vidx_jump >> VSO
                        if (vidx_jump := struct.unpack_from(">I", self.layers[nodes[0]], (nidx * NS + VOF))[0]) & JSFT
                        # 直接&JSFT是因为布尔判断不需要移位到0x01
                        else nidx
                    )
                    for nidx in samples
                ]
            )
            if tails[-1] >= nodes[1] + nodes[2]:
                tails = tails[:-1]

            statistic: defaultdict[int, list[int]] = defaultdict(list[int])
            weight: dict[int, float] = {}
            for sample, tail in zip(samples, tails):
                if tail != VDFT:
                    statistic[tail].append(sample)

            if len(statistic) < num:
                if trial >= nodes[2]:
                    if exact:
                        raise SourceDrainedError
                    return [
                        marisa.restore_key(struct.unpack_from(">I", self.layers[nodes[0]], (idx * NS + VOF))[0] >> VSO)
                        for idx in statistic.keys()
                    ]
                trial = int(trial * 1.5)  # 此处*1.5要放在短路之后，为了保证极限情况下不会有漏抽
                continue

            for key, vals in statistic.items():
                dmax = key - min(vals)
                cnt = len(vals)
                weight[key] = 1 / ((dmax + 1e-6) * (1 + 1 / cnt))
            unique, weights = np.array(list(weight.keys()), uint32), np.array(list(weight.values()), np.float64)
            selected = np.random.choice(unique, num, p=weights / weights.sum(), replace=False)
            return [marisa.restore_key(struct.unpack_from(">I", self.layers[nodes[0]], (idx * NS + VOF))[0] >> VSO) for idx in selected]

    def extract_fast(self, nodes: Tnodes, num: int, exact: bool = False) -> list[str]:
        # 牺牲采样质量换取高（甚至可能是极高）的速度增益。
        # 虽然说采样质量确实不是什么至关紧要的指标，但关键是如果纯随机的话概率对比度最高能有几百万，这真的是能接受的吗……
        # 「凡事皆有其代价。凡事皆然。」
        try:
            marisa = self.lval[nodes[0]]
        except IndexError:
            if exact:
                raise
            return []

        samples = random.choices(range(nodes[2]), k=num)

        chosen = [
            (vidx_jump >> VSO if (vidx_jump := struct.unpack_from(">I", self.layers[nodes[0]], (nidx * NS + VOF))[0]) & JSFT else nidx)
            for nidx in samples
        ]

        return [marisa.restore_key(idx) for idx in chosen]

    def extract_batch(self, segs: Iterable[Tnode | Tnodes], num: int, exact: bool = False) -> list[str]:
        src = self.seg_merge(segs)
        if not src:
            raise SourceEmptyError
        weights = [nodes[2] for nodes in src]
        selected = random.choices(src, weights, k=num)
        count = Counter(selected)
        return list(chain(*(self.extract(nodes, count[nodes], exact) for nodes in count)))

    def extract_unique_batch(self, src: Iterable[Tnodes | Tnode], num: int, exact: bool = False) -> list[str]:
        # 此方法适用的值格式: SSV（空格分割）的一键多值
        nodes = self.seg_merge(src)
        trial = int(num * (1 + SSVR * 3) + 1)  # 使用SSVR的3倍来保证尽可能不要重试
        pool_size = sum(n[2] for n in nodes)
        while True:
            raw = self.extract_batch(nodes, trial)
            flattened: chain[str] = chain(*(item.split(" ") for item in raw))
            # TODO: 等到3.15 PEP 798正式落实，就可以直接写[*item.split(" ") for item in raw]了
            counter = Counter(flattened)
            unique = list(counter.keys())
            if len(unique) < num:
                if trial >= pool_size:
                    # 那就是实在不够了
                    if exact:
                        raise SourceDrainedError
                    random.shuffle(unique)
                    return list(unique)
                trial = int(trial * 1.5)
                continue  # 抽少了说是
            weight = np.array(list(counter.values()), np.float64)
            weight = 1 / weight  # 取倒数以实现逆概率校正
            weight /= weight.sum()  # 归一化。鬼才知道为什么np不内置归一化
            return list(np.random.choice(unique, num, p=weight, replace=False))

    def extract_unique(self, nodes: Tnodes | Tnode, num: int, exact: bool = False) -> list[str]:
        return self.extract_unique_batch([nodes], num, exact)

    def get_val(self, node: Tnode) -> str | None:
        lidx, nidx = node
        try:
            marisa = self.lval[lidx]
        except KeyError:
            return None
        val_jmp_mdpt = struct.unpack_from(">I", self.layers[lidx], nidx * NS + VOF)[0]
        vidx = val_jmp_mdpt >> VSO
        if val_jmp_mdpt & JSFT:
            return None
        return marisa.restore_key(vidx)

    def rand_nodes(self, num: int) -> Generator[Tnode]:
        for _ in range(num):
            lidx = random.randrange(0, len(self.layers))
            llen = len(self.layers[lidx]) // NS
            nidx = random.randrange(0, llen)
            yield lidx, nidx

    def rand_val_at_layer(self, lidx: int, num: int, exact: bool = False) -> list[str]:
        try:
            marisa = self.lval[lidx]
        except KeyError:
            if exact:
                raise
            return []
        if len(marisa) < num:
            if exact:
                raise SourceDrainedError
            return list(marisa.iterkeys())
        choices = random.sample(range(len(marisa)), k=num)
        return [marisa.restore_key(idx) for idx in choices]

    def rand_unique_val_at_layer(self, lidx: int, num: int, exact: bool = False) -> list[str]:
        try:
            marisa = self.lval[lidx]
        except KeyError:
            if exact:
                raise
            return []

        trial = int(num * (1 + SSVR * 3) + 1)

        while True:
            if len(marisa) <= trial:
                choices = range(len(marisa))
            else:
                choices = np.random.choice(len(marisa), trial, replace=False)

            raw_values = [marisa.restore_key(idx) for idx in choices]
            flattened = chain(*(item.split(" ") for item in raw_values))

            counter = Counter(flattened)
            unique = list(counter.keys())

            if len(unique) < num:
                if trial >= len(marisa):
                    if exact:
                        raise SourceDrainedError
                    return list(unique)
                trial = int(trial * 1.5)
                continue

            weights = np.array(list(counter.values()), np.float64)
            weights = 1 / weights
            weights /= weights.sum()

            return list(np.random.choice(unique, num, p=weights, replace=False))

    def rand_val(self, num: int, exact: bool = False) -> list[str]:
        lvls = list(self.lval.keys())
        lengths = list(map(len, self.lval.values()))
        weights = np.array(lengths, np.float32)
        weights /= weights.sum()
        selected = np.random.choice(lvls, int(num * 1.02), p=weights)
        count = Counter(selected)
        ret = list(chain(*(self.rand_val_at_layer(lidx, ppl, exact) for lidx, ppl in count.items())))
        if len(ret) < num:
            if exact:
                raise SourceDrainedError
            return ret
        return ret[:num]

    def rand_unique_val(self, num: int, exact: bool = False) -> list[str]:
        trial = int(num * (1 + SSVR * 3) + 1)

        while True:
            raw = self.rand_val(trial, exact=False)

            flattened: chain[str] = chain(*(item.split(" ") for item in raw))
            counter = Counter(flattened)
            unique = list(counter.keys())

            if len(unique) < num:
                total_nodes = sum(len(marisa) for marisa in self.lval.values())
                if trial >= total_nodes:
                    if exact:
                        raise SourceDrainedError
                    return list(unique)
                trial = int(trial * 1.5)
                continue

            weights = np.array(list(counter.values()), np.float64)
            weights = 1 / weights
            weights /= weights.sum()

            return list(np.random.choice(unique, num, p=weights, replace=False))

    def navigate(self, key: bytes) -> Tnode:  # 返回的是(层数, 索引)二元组。索引为编号索引nidx不是内存索引nmidx。
        if not 0 < len(key) <= len(self.layers):
            raise ValueError
        parent = PDFT
        for i, char in enumerate(key):
            layer = memoryview(self.layers[i])
            head = ((parent << PSO) | char).to_bytes(4, "big")
            pos = headfindHOE(layer, head)
            if pos == -1:
                raise IndexError
            parent = pos
        return len(key) - 1, parent

    def search(
        self, key: Sequence[bytes | list[tuple[uint8, uint8]] | None], leaf_only: bool = False, src: list[Tnode] | None = None
    ) -> Generator[Tnode, None, None]:
        if not 0 < len(key) <= len(self.layers):
            return
        queue: deque[tuple[int, uint24, int]] = deque()  # lidx, nidx, kidx

        if not src:
            try:
                prefix_byte = next(i for i in range(len(key)) if not (isinstance(key[i], bytes) and len(cast(bytes, key[i])) == 1))
                if prefix_byte:
                    src = [self.navigate(b"".join(cast(list[bytes], key[:prefix_byte])))]
                    key = key[prefix_byte:]
            except StopIteration:
                yield self.navigate(b"".join(cast(list[bytes], key)))
                return

        try:
            prefix_nones = next(i for i in range(len(key)) if key[i] is not None)
        except StopIteration:
            # 全是None是什么鬼。
            if src:
                yield from chain(*(self.expand(self.get_successor(a_src, a_src[1] + len(key))) for a_src in src))
            elif self.layers:
                yield from ((0, i) for i in range(len(self.layers[len(key) - 1]) // NS))
            return

        if src:
            queue.extend(
                (*succ, prefix_nones) for src_node in src for succ in self.expand(self.get_successor(src_node, src_node[0] + prefix_nones + 1))
            )
        elif self.layers:
            queue.extend((prefix_nones, i, prefix_nones) for i in range(len(self.layers[prefix_nones]) // NS))

        lkm1 = len(key) - 1  # Len Key Minus 1

        trailing_nones_start = next((i for i in range(lkm1, -1, -1) if key[i] is not None), -1)

        while queue:
            lidx, nidx, kidx = queue.popleft()
            char, cidx_clen, val_jmp_mdpt = struct.unpack_from(">BII", self.layers[lidx], nidx * NS + POE)
            char: uint8
            cidx_clen: uint32
            val_jmp_mdpt: uint32

            if (
                (kidx > trailing_nones_start)
                or ((pattern := key[kidx]) is None)
                or ((char in pattern) if isinstance(pattern, bytes) else any(lower <= char <= upper for lower, upper in pattern))
            ):
                cidx, clen, jump, mdpt = cidx_clen >> CSO, cidx_clen & LMS, val_jmp_mdpt >> JSO & JMS, val_jmp_mdpt & MMS
                if kidx == lkm1:
                    if not (leaf_only and jump):
                        yield (lidx, nidx)
                elif clen and (mdpt >= min(MDPTH, lkm1 - kidx)):
                    queue.extend((lidx + 1, cidx + i, kidx + 1) for i in range(clen))

    def rebuild_path(self, node: Tnode, min_lidx: int = 0) -> Iterator[Tnode]:
        path = [node]
        layer_idx, nidx = node
        while layer_idx > min_lidx:
            pidx: int = struct.unpack_from(">I", self.layers[layer_idx], nidx * NS)[0] >> PSO
            path.append((layer_idx - 1, pidx))
            layer_idx -= 1
            nidx = pidx
        return reversed(path)

    def get_ancestor(self, node: Tnode, lidx: int) -> Tnode:
        layer_idx, nidx = node
        while layer_idx > lidx:
            pidx: int = struct.unpack_from(">I", self.layers[layer_idx], nidx * NS)[0] >> PSO
            layer_idx -= 1
            nidx = pidx
        return (layer_idx, nidx)

    def get_successor(self, node: Tnode, lidx: int, default: Any = None) -> Tnodes:
        mlen = struct.unpack_from(">B", self.layers[node[0]], node[1] * NS + MOF)[0]
        if (mlen & MMS) < min(MDPTH, lidx - node[0]):
            if default is not None:
                return default
            raise IndexError
        lo, hi = 0, len(self.layers[lidx]) // NS
        plidx, pnidx = node
        while lo < hi:
            mid = (lo + hi) // 2
            if self.get_ancestor((lidx, mid), plidx)[1] < pnidx:
                lo = mid + 1
            else:
                hi = mid
        lo_l = lo
        hi = len(self.layers[lidx]) // NS
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.get_ancestor((lidx, mid), plidx)[1] <= pnidx:
                lo = mid
            else:
                hi = mid - 1

        return (lidx, lo_l, lo - lo_l + 1)

    def get_successor_tnodes(self, node: Tnodes, lidx: int, default: Any = None) -> Tnodes:
        mlen = struct.unpack_from(">B", self.layers[node[0]], node[1] * NS + MOF)[0]
        if (mlen & MMS) < min(MDPTH, lidx - node[0]):
            if default is not None:
                return default
            raise IndexError
        lo, hi = 0, len(self.layers[lidx]) // NS
        plidx, pnidxl, pnidxs = node
        pnidxr = pnidxl + pnidxs - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if self.get_ancestor((lidx, mid), plidx)[1] < pnidxl:
                lo = mid + 1
            else:
                hi = mid
        lo_l = lo
        hi = len(self.layers[lidx]) // NS
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.get_ancestor((lidx, mid), plidx)[1] <= pnidxr:
                lo = mid
            else:
                hi = mid - 1

        return (lidx, lo_l, lo - lo_l + 1)

    def get_successor_batch(self, segs: Iterable[Tnode], lidx: int) -> list[Tnodes]:
        return self.seg_merge(result for node in segs if (result := self.get_successor(node, lidx, default=0)))

    def get_successor_batch_tnodes(self, segs: Iterable[Tnodes], lidx: int) -> list[Tnodes]:
        return self.seg_merge(result for node in segs if (result := self.get_successor_tnodes(node, lidx, default=0)))

    def get_successor_from(self, node: Tnode, lidx: int, leaf_only: bool = False) -> Generator[Tnodes, None, None]:
        for lvl_idx in range(lidx, len(self.layers)):
            if leaf_only and lvl_idx not in self.lval:
                continue
            with suppress(IndexError):
                yield self.get_successor(node, lvl_idx)

    def get_successor_from_batch(self, segs: Iterable[Tnode], lidx: int, leaf_only: bool = False) -> Generator[Tnodes, None, None]:
        src = self.seg_merge(segs)
        for lvl_idx in range(lidx, len(self.layers)):
            if leaf_only and lvl_idx not in self.lval:
                continue
            # with suppress(IndexError):
            dbg = self.get_successor_batch_tnodes(src, lvl_idx)
            print(f"DEBUG: {dbg=} {lvl_idx=}")
            yield from dbg

    def node_data(self, node: tuple[int, uint24]) -> TnodeData:
        lidx, nidx = node
        return MLTrie._unpack(self.layers[lidx][nidx * NS : nidx * NS + NS])

    def node_val(self, node: tuple[int, uint24]) -> str:
        lidx, nidx = node
        return self.lval[lidx].restore_key(struct.unpack_from(">I", self.layers[lidx], nidx * NS + VOF)[0] >> VSO)

    def node_char(self, node: tuple[int, uint24]) -> str:
        lidx, nidx = node
        return chr(self.layers[lidx][nidx * NS + 3])

    def to_val(self, upstream: Iterable[tuple[int, uint24]]) -> list[str]:
        return list(map(self.node_val, upstream))

    def to_char(self, upstream: Iterable[tuple[int, uint24]]) -> list[str]:
        return list(map(self.node_char, upstream))

    def check_integrity(self):
        if any((len(layer) >= MLLEN * NS) or (len(layer) % NS) for layer in self.layers) or any(len(lv) >= MVCNT for lv in self.lval.values()):
            raise OverflowError  # 索引是3B，能表示的最大索引是16777215，长度对应的就是201326580

    def get_memusage(self) -> int:
        # 注意: 本函数给出的数值是有低估的（来源于marisa的不准确以及一些杂七杂八的额外开销）
        layer_usage = sum(len(layer) for layer in self.layers) + 33 * len(self.layers)  # 33: bytes的元数据大小
        marisa_usage = sum(len(pickle.dumps(marisa)) for marisa in self.lval.values())  # 可能慢的依托构史但是没有别的方法了QwQ
        return layer_usage + marisa_usage

    def get_value_count(self) -> int:
        return sum(len(lv) for lv in self.lval.values())

    def __contains__(self, key: str) -> bool:
        return key in self.lval.get(len(key) - 1, ())

"""Original GLB morphs mapped to runtime skins; physics and source files unchanged.

Base mesh/art and morphs: 木子恶魔. Runtime mapping and blending: OpenDafeiyu.
"""
import hashlib
import json
import struct
import io
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial import cKDTree
from .bluefatfish import DELIVERY


def adapt_alpha_decals(model, xml, assets, subdivisions=64):
    """Convert only the alpha decal's transparent regions to absent triangles.

    MuJoCo's built-in GL renderer does not alpha-test this skin's RGBA texture.
    Preserve its original artwork and derive a cutout skin with barycentric
    morph correspondence. All opaque character meshes and collision shapes stay.
    """
    from PIL import Image
    from .bluefatfish import values
    si=model.skin('skin_5_1').id; start=int(model.skin_vertadr[si]); count=int(model.skin_vertnum[si])
    vs=model.skin_vert[start:start+count];uv=model.skin_texcoord[int(model.skin_texcoordadr[si]):][:count]
    faces=model.skin_face[int(model.skin_faceadr[si]):][:int(model.skin_facenum[si])//2]
    alpha=np.asarray(Image.open(io.BytesIO(assets['textures/5_expression.png'])))[:,:,3]
    grid=[];lookup={}
    for i in range(subdivisions+1):
        for j in range(subdivisions+1-i):
            lookup[i,j]=len(grid);grid.append([1-(i+j)/subdivisions,i/subdivisions,j/subdivisions])
    grid=np.array(grid); triangles=[]
    for i in range(subdivisions):
        for j in range(subdivisions-i):
            triangles.append([lookup[i,j],lookup[i+1,j],lookup[i,j+1]])
            if i+j<subdivisions-1:triangles.append([lookup[i+1,j],lookup[i+1,j+1],lookup[i,j+1]])
    triangles=np.array(triangles)
    vertices=[];texcoords=[];indices=[];weights=[];newfaces=[];offset=0
    for face in faces:
        tuv=grid@uv[face];centroid=tuv[triangles].mean(axis=1)
        x=np.clip((centroid[:,0]*alpha.shape[1]).astype(int),0,alpha.shape[1]-1)
        # The received MuJoCo skin already uses the PNG row convention.
        y=np.clip((centroid[:,1]*alpha.shape[0]).astype(int),0,alpha.shape[0]-1)
        kept=triangles[alpha[y,x]>=128]
        if not len(kept):continue
        used,inv=np.unique(kept,return_inverse=True)
        vertices.extend(grid[used]@vs[face]);texcoords.extend(tuv[used]);indices.extend(np.tile(face,(len(used),1)));weights.extend(grid[used])
        newfaces.extend(inv.reshape(-1,3)+offset);offset+=len(used)
    vertices=np.array(vertices);texcoords=np.array(texcoords);indices=np.array(indices);weights=np.array(weights);newfaces=np.array(newfaces)
    n=len(vertices)
    vertices=np.concatenate([vertices,vertices]);texcoords=np.concatenate([texcoords,texcoords])
    newfaces=np.concatenate([newfaces,newfaces[:,::-1]+n]);indices=np.concatenate([indices,indices]);weights=np.concatenate([weights,weights])
    tree=ET.fromstring(xml);skin=tree.find(".//skin[@name='skin_5_1']")
    skin.attrib.pop('file');skin.set('vertex',values(vertices));skin.set('texcoord',values(texcoords));skin.set('face',' '.join(map(str,newfaces.ravel())))
    for bi in range(int(model.skin_boneadr[si]),int(model.skin_boneadr[si]+model.skin_bonenum[si])):
        source_weights=np.zeros(count);adr=int(model.skin_bonevertadr[bi]);num=int(model.skin_bonevertnum[bi])
        source_weights[model.skin_bonevertid[adr:adr+num]]=model.skin_bonevertweight[adr:adr+num]
        target_weights=(source_weights[indices]*weights).sum(axis=1);ids=np.flatnonzero(target_weights>1e-8)
        ET.SubElement(skin,'bone',body=model.body(int(model.skin_bonebodyid[bi])).name,
            bindpos=values(model.skin_bonebindpos[bi]),bindquat=values(model.skin_bonebindquat[bi]),
            vertid=' '.join(map(str,ids)),vertweight=values(target_weights[ids]))
    mat=tree.find("asset/material[@name='mat_5']")
    for child in list(mat):mat.remove(child)
    mat.set('texture','tex_5')
    data=io.BytesIO();np.savez_compressed(data,indices=indices,weights=weights,original=vs.copy())
    assets=assets.copy();assets['expression-cutout-map.npz']=data.getvalue()
    return ET.tostring(tree,encoding='unicode'),assets,dict(source_skin='skin_5_1',subdivisions=subdivisions,
        vertices=len(vertices),faces=len(newfaces),alpha_threshold=128,
        mapping_asset='expression-cutout-map.npz',scope='Derived alpha cutout for visual decals only')


class Expressions:
    def __init__(self, model, delivery=DELIVERY, mapping=None):
        self.model = model
        raw = (delivery/'BlueFatFish_Visual.glb').read_bytes()
        size = struct.unpack_from('<I', raw, 12)[0]
        gltf = json.loads(raw[20:20+size]); buffer = raw[28+size:]
        def accessor(index):
            a = gltf['accessors'][index]
            if a['componentType'] != 5126 or a['type'] != 'VEC3':
                raise ValueError('Unsupported morph accessor')
            result=np.zeros((a['count'],3),dtype=np.float32)
            if 'bufferView' in a:
                v = gltf['bufferViews'][a['bufferView']]
                result=np.ndarray((a['count'], 3), dtype='<f4', buffer=buffer,
                    offset=v.get('byteOffset',0)+a.get('byteOffset',0),
                    strides=(v.get('byteStride',12),4)).copy()
            if 'sparse' in a:
                sp=a['sparse']; ix=sp['indices']; val=sp['values']
                vi=gltf['bufferViews'][ix['bufferView']]; vv=gltf['bufferViews'][val['bufferView']]
                ids=np.frombuffer(buffer,dtype={5121:'u1',5123:'<u2',5125:'<u4'}[ix['componentType']],count=sp['count'],offset=vi.get('byteOffset',0)+ix.get('byteOffset',0))
                result[ids]=np.frombuffer(buffer,dtype='<f4',count=3*sp['count'],offset=vv.get('byteOffset',0)+val.get('byteOffset',0)).reshape(-1,3)
            return result[:,[0,2,1]] * [1,-1,1]
        self.neutral = model.skin_vert.copy()
        cutout=np.load(io.BytesIO(mapping)) if mapping is not None else None
        self.deltas = {}; self.weights = {}; errors = []
        for mi, mesh in enumerate(gltf['meshes']):
            names = mesh.get('extras',{}).get('targetNames',[])
            if not names: continue
            for pi, primitive in enumerate(mesh['primitives']):
                skin = model.skin(f'skin_{mi}_{pi}').id
                start = model.skin_vertadr[skin]; count = model.skin_vertnum[skin]
                source = accessor(primitive['attributes']['POSITION'])
                reference=cutout['original'] if cutout is not None and mi==5 and pi==1 else self.neutral[start:start+count]
                distance, vertex_map = cKDTree(source).query(reference)
                if distance.max() > 1e-6:
                    raise ValueError('Morph/skin correspondence exceeds 1 micrometre')
                errors.append(float(distance.max()))
                for name, target in zip(names, primitive['targets']):
                    delta = accessor(target['POSITION'])
                    # UV seam duplicates must have the same displacement. A nearest
                    # vertex cannot silently select a different coincident shape.
                    groups = cKDTree(source).query_pairs(1e-8, output_type='ndarray')
                    if len(groups) and np.abs(delta[groups[:,0]]-delta[groups[:,1]]).max()>1e-6:
                        raise ValueError('Ambiguous coincident morph vertices')
                    full = self.deltas.setdefault(name, np.zeros_like(self.neutral))
                    mapped=delta[vertex_map]
                    if cutout is not None and mi==5 and pi==1:
                        mapped=(mapped[cutout['indices']]*cutout['weights'][:,:,None]).sum(axis=1)
                    full[start:start+count] = mapped
        self.report = dict(source_sha256=hashlib.sha256(raw).hexdigest(),
            names=list(self.deltas), max_mapping_error_m=max(errors),
            method='Original GLB morph delta in source bind frame; validated nearest-vertex mapping, both skin sides',
            scope='Visual skin vertices only; no collision, mass, joint, qpos or force changes')
        self._emotion='neutral'
        self._transition_time=0.
        self._from_weights={}
        self._semantic_weights={}
        self.report['runtime_blending']='VD2.1: 280 ms semantic transition; relaxed uses natural blink; original comic relaxed remains available through apply.'

    def apply(self, weights):
        if any(k not in self.deltas for k in weights): raise ValueError('Unknown expression')
        self.weights = {k:float(np.clip(v,0,1)) for k,v in weights.items() if v>0}
        self.model.skin_vert[:] = self.neutral
        for k,v in self.weights.items(): self.model.skin_vert[:] += v*self.deltas[k]

    def update(self, time, emotion='neutral', talking=False):
        # The source 'relaxed' is a comic replacement eye: its intermediate
        # morph hides iris/lashes before the replacement decal reaches the face.
        # Keep that original morph intact for explicit apply/replays; map the
        # normal relaxed state to the original continuous eyelid closure instead.
        targets={'neutral':{},'happy':{'happy':.85},'angry':{'angry':.85},
                 'sad':{'sad':.85},'relaxed':{'blink':1.},
                 'focus':{'blink':.12},'effort':{'blink':.65,'aa':.08},'delight':{'happy':.55}}
        targets.update(getattr(self,'semantic_targets',{}))
        if emotion not in targets:raise ValueError('Unknown emotion')
        if emotion!=self._emotion:
            self._from_weights=self._semantic_weights.copy()
            self._transition_time=time;self._emotion=emotion
        u=float(np.clip((time-self._transition_time)/.28,0,1));u=u*u*(3-2*u)
        target=targets[emotion]
        self._semantic_weights={k:(1-u)*self._from_weights.get(k,0)+u*target.get(k,0)
                                for k in self._from_weights.keys()|target.keys()}
        weights=self._semantic_weights.copy()
        # Asymmetric eyelid timing: 80 ms close, 30 ms hold, 140 ms reopen.
        # Only neutral uses autonomous blinks; expression changes never stack
        # another blink over an existing closed-eye expression.
        phase=(time-3.2)%3.7
        if emotion=='neutral' and u>=1 and phase<.25:
            v=phase/.08 if phase<.08 else (1. if phase<.11 else 1-(phase-.11)/.14)
            v=float(np.clip(v,0,1));weights['blink']=v*v*(3-2*v)
        if talking: weights['aa'] = .35*(.5+.5*np.sin(2*np.pi*3*time))
        self.apply(weights)

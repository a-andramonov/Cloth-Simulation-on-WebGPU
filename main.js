
const clothRes = 20; 
const canvas = document.getElementById('webgpu-canvas');

function createCloth(resolution, size) {
    const particles = new Float32Array((resolution+1) * (resolution+1) * 8);
    let pIdx = 0;
    for(let r = 0; r <= resolution; r++) {
        for(let c = 0; c <= resolution; c++) {
            let x = (c / resolution) * size - size / 2;
            let y = 0.0;
            let z = (r / resolution) * size - size / 2;

            if ((c === 0 && r === 0) || (c === resolution && r === resolution) || (c === 0 && r === resolution) || (c === resolution && r === 0)) {
                particles.set([x, y, z, 1.0, x, y, z, 1.0], pIdx*8);}
            else if (c === resolution / 2 && r === resolution / 2) {particles.set([x, y, z, 2.0, x, y, z, 2.0], pIdx*8);}
            else {particles.set([x, y, z, 0.0, x, y, z, 0.0], pIdx*8);}
            pIdx++;
        }
    }

    const indices = new Uint32Array(resolution * resolution * 6);
    let iIdx = 0;
    for (let r = 0; r < resolution; r++) {
        for (let c = 0; c < resolution; c++) {
        const i = r * (resolution + 1) + c;
        indices.set([i, i + resolution + 1, i + 1, i + 1, i + resolution + 1, i + resolution + 2], iIdx);
        iIdx += 6;
    }}
    return {particles, indices};
}


function getMVPMatrix() {
    const aspect = canvas.width / canvas.height;
    const f = 1.0 / Math.tan(Math.PI / 8); // Поле зрения 45 градусов
    const proj = [f/aspect,0,0,0, 0,f,0,0, 0,0,-1.001,-1, 0,0,-0.1001,0]; // Проекция
    
    // Камера висит сверху-сбоку (eye) и смотрит в центр (target)
    const eye = [7.0, 2.3, 2.5], target = [0, 0, 0], up = [0, 1, 0];
    
    // Вычисляем оси камеры 
    let z = [eye[0]-target[0], eye[1]-target[1], eye[2]-target[2]];
    let len = Math.hypot(z[0], z[1], z[2]); z = z.map(v => v/len);
    
    let x = [up[1]*z[2] - up[2]*z[1], up[2]*z[0] - up[0]*z[2], up[0]*z[1] - up[1]*z[0]];
    len = Math.hypot(x[0], x[1], x[2]); x = x.map(v => v/len);
    
    let y = [z[1]*x[2] - z[2]*x[1], z[2]*x[0] - z[0]*x[2], z[0]*x[1] - z[1]*x[0]];
    
    const view = [ x[0],y[0],z[0],0, x[1],y[1],z[1],0, x[2],y[2],z[2],0, -(x[0]*eye[0]+x[1]*eye[1]+x[2]*eye[2]), -(y[0]*eye[0]+y[1]*eye[1]+y[2]*eye[2]), -(z[0]*eye[0]+z[1]*eye[1]+z[2]*eye[2]), 1 ];
    
    // Умножаем Projection * View
    const mvp = new Float32Array(16);
    for (let c = 0; c < 4; c++) {
        for (let r = 0; r < 4; r++) {
            mvp[c*4+r] = proj[0*4+r]*view[c*4+0] + proj[1*4+r]*view[c*4+1] + proj[2*4+r]*view[c*4+2] + proj[3*4+r]*view[c*4+3];
        }
    }
    return mvp;
}

async function init() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;

    if (!navigator.gpu) return alert('WebGPU не поддерживается!');

    const adapter = await navigator.gpu.requestAdapter();
    const device = await adapter.requestDevice();
    const context = canvas.getContext('webgpu');
    const canvasFormat = navigator.gpu.getPreferredCanvasFormat();
    context.configure({ device, format: canvasFormat, alphaMode: 'premultiplied' });

    const clothData = createCloth(clothRes, 2.0);

    const shaderWGSL = `
        struct Uniforms {
        mvp: mat4x4<f32>,};
        @group(0) @binding(0) var<uniform> uniforms: Uniforms;

        struct VertexOutput {
            @builtin(position) position: vec4<f32>,
            @location(0) color: vec3<f32>,
            @location(1) world_pos: vec3<f32>,
        };

        @vertex
        fn vs_main(@location(0) pos_type: vec4<f32>, @location(1) old_pos: vec4<f32>) -> VertexOutput {
            let w = pos_type.w;
            var color = vec3<f32>(0.0, 0.0, 0.0);
            if (w == 0.0) {color = vec3<f32>(0.5, 0.5, 0.5);}
            if (w == 1.0) {color = vec3<f32>(1.0, 0.0, 0.0);}
            if (w == 2.0) {color = vec3<f32>(0.0, 0.0, 1.0);}
            return VertexOutput(uniforms.mvp * vec4<f32>(pos_type.xyz, 1.0), color, pos_type.xyz);
        }

        @fragment
        fn fs_main(@location(0) color: vec3<f32>, @location(1) world_pos: vec3<f32>) -> @location(0) vec4<f32> {
            let gridX = abs(fract(world_pos.x * 10.0) - 0.5) < 0.05;
            let gridZ = abs(fract(world_pos.z * 10.0) - 0.5) < 0.05;
            let diagThickness = 0.05 * 1.414; 
            let diag1 = abs(fract(world_pos.x * 10.0) - fract(world_pos.z * 10.0)) < diagThickness;
            if (gridX || gridZ || diag1) { return vec4<f32>(0.1, 0.1, 0.1, 1.0); }
            return vec4<f32>(color, 1.0);
        }
    `;

    const compute_shader = `
        struct Particle {
            pos: vec4<f32>,
            old_pos: vec4<f32>
        };

        struct Particles {
        particles: array<Particle>,};
        @group(0) @binding(1) var<storage, read_write> particles: Particles;
        
        struct Params { time: f32, gravity: f32, res: f32, restDist: f32, frequency: f32, amplitude: f32 };
        @group(0) @binding(0) var<uniform> params: Params;

        @compute @workgroup_size(64)
        fn integrate_main(@builtin(global_invocation_id) id: vec3<u32>) {
            let total = u32(params.res + 1.0) * u32(params.res + 1.0);
            let index = id.x;
            if (index >= total) { return; }

            var p = particles.particles[index];
            let ptype = p.pos.w;

            if ptype == 1 {return;}

            if ptype == 2 {
                p.old_pos = p.pos;
                p.pos.y = sin(params.time * params.frequency) * params.amplitude;
                particles.particles[index] = p;
            }

            if ptype == 0 {
                let dampling = 0.99;
                var velocity = (p.pos.xyz - p.old_pos.xyz) * dampling;
                p.old_pos = p.pos;
                p.pos = vec4<f32>(p.pos.xyz + velocity, p.pos.w);
                if (params.gravity > 0.5) {p.pos.y -= 0.003;}
                particles.particles[index] = p;
            }

            }
        
    `;

    const pbd_shader = `
    struct Particle {
        pos: vec4<f32>,
        old_pos: vec4<f32>
    };

    struct Particles {
        particles: array<Particle>,
    };

    @group(0) @binding(1) var<storage, read_write> particles: Particles;

    struct Params { time: f32, gravity: f32, res: f32, restDist: f32 };
    @group(0) @binding(0) var<uniform> params: Params;

    fn get_spring(pos: vec3<f32>, neighbor_index: u32, target_dist: f32) -> vec4<f32> {
        let d = pos - particles.particles[neighbor_index].pos.xyz;
        let l = length(d);
        let error = l - target_dist;

        let epsilon = 0.0001;
        if l > epsilon {
            let correction = (d / l) * -error;
            return vec4<f32>(correction, 1.0);
        }
        
        return vec4<f32>(0.0);
    }

    @compute @workgroup_size(64)
    fn solve_main(@builtin(global_invocation_id) id: vec3<u32>) {

    let res = u32(params.res);
    let total = u32(res + 1u) * u32(res + 1u);
    let index = id.x;

    if (index >= total) { return; }

    var p = particles.particles[index];
    let ptype = p.pos.w;

    if ptype != 0 {return;}

    let r = index / (res + 1u);
    let c = index - r * (res + 1u);

    var w = 0.0;
    var corr = vec3<f32>(0.0);

    if (c > 0) {
        let f =  get_spring(p.pos.xyz, index - 1, params.restDist);
        corr += f.xyz;
        w += f.w;
    }
    if (c < res) {
        let f = get_spring(p.pos.xyz, index + 1, params.restDist);
        corr += f.xyz;
        w += f.w;
    }
    if (r > 0) {
        let f = get_spring(p.pos.xyz, index - res - 1, params.restDist);
        corr += f.xyz;
        w += f.w;
    }
    if (r < res) {
        let f = get_spring(p.pos.xyz, index + res + 1, params.restDist);
        corr += f.xyz;
        w += f.w;
    }
    if (c != 0 && r != 0) {
        let f = get_spring(p.pos.xyz, index - res - 2, params.restDist * 1.4142);
        corr += f.xyz;
        w += f.w;
    }
    if (c != res && r != 0) {
        let f = get_spring(p.pos.xyz, index - res, params.restDist * 1.4142);
        corr += f.xyz;
        w += f.w;
    }
    if (c != res && r != res) {
        let f = get_spring(p.pos.xyz, index + res + 2, params.restDist * 1.4142);
        corr += f.xyz;
        w += f.w;
    }
    if (c != 0 && r != res) {
        let f = get_spring(p.pos.xyz, index + res, params.restDist * 1.4142);
        corr += f.xyz;
        w += f.w;
    }

    if (w > 0.0) {
        let temp = p.pos.xyz + corr/w;
        p.pos = vec4<f32>(temp, p.pos.w);
    }

    particles.particles[index] = p;

}`;

    const shaderModule = device.createShaderModule({ code: shaderWGSL });
    const compute_shaderModule = device.createShaderModule({code: compute_shader});
    const pbd_shaderModule = device.createShaderModule({code: pbd_shader});
    
    const particleBuffer = device.createBuffer({
        usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE,
        size: clothData.particles.byteLength,
    });
    device.queue.writeBuffer(particleBuffer, 0, clothData.particles);
    const indexBuffer = device.createBuffer({
        usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
        size: clothData.indices.byteLength,
    });
    device.queue.writeBuffer(indexBuffer, 0, clothData.indices);
    const uniformBuffer = device.createBuffer({
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        size: 64,
    });
    device.queue.writeBuffer(uniformBuffer, 0, getMVPMatrix());
    const simParamsBuffer = device.createBuffer({
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        size: 24,
    });

    
    const pipeline = device.createRenderPipeline({
        layout: "auto",
        vertex: {
            module: shaderModule,
            entryPoint: "vs_main",
            buffers: [{ arrayStride: 32, attributes: [
       { shaderLocation: 0, offset: 0,  format: 'float32x4' },
       { shaderLocation: 1, offset: 16, format: 'float32x4' }
        ]}],},

        primitive: { topology: 'triangle-list', cullMode: 'none' },        
        fragment: {
            module: shaderModule,
            entryPoint: 'fs_main',
            targets: [{format: canvasFormat}],
        }  
    });

    const compute_pipeline = device.createComputePipeline({
        layout: "auto",
        compute: { module: compute_shaderModule, entryPoint: 'integrate_main' },
    });

    const pbd_pipeline = device.createComputePipeline({
        layout: "auto",
        compute: { module: pbd_shaderModule, entryPoint: 'solve_main' },
    });

    const pbdBindGroup = device.createBindGroup({
        layout: pbd_pipeline.getBindGroupLayout(0),
        entries: [{binding: 1,
        resource: {
            buffer: particleBuffer,
            offset: 0,
            size: (clothRes+1) * (clothRes+1) * 8 * 4,
        }
        },
        {binding: 0,
        resource: {
            buffer: simParamsBuffer,
            offset: 0,
            size: 16,
        }

        }
    ]
    });

    const uniformBindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [{binding: 0,
        resource: {
            buffer: uniformBuffer,
            offset: 0,
            size: 64,
        }
        }]
    });

    const computeBindGroup = device.createBindGroup({
        layout: compute_pipeline.getBindGroupLayout(0),
        entries: [{binding: 1,
        resource: {
            buffer: particleBuffer,
            offset: 0,
            size: (clothRes+1) * (clothRes+1) * 8 * 4,
        }
        },
        {binding: 0,
        resource: {
            buffer: simParamsBuffer,
            offset: 0,
            size: 24,
        }

        }
    ]
    });

    function render() {
        const commandEncoder = device.createCommandEncoder();
        const workgroupCount = Math.ceil((clothRes + 1) * (clothRes + 1) / 64);
        const computePass = commandEncoder.beginComputePass();
        computePass.setPipeline(compute_pipeline);
        computePass.setBindGroup(0, computeBindGroup);
        computePass.dispatchWorkgroups(workgroupCount);
        
        for (let i = 0; i < 32; i++) {
            computePass.setPipeline(pbd_pipeline);
            computePass.setBindGroup(0, pbdBindGroup);
            computePass.dispatchWorkgroups(workgroupCount);
        }
        computePass.end();

        const renderPass = commandEncoder.beginRenderPass({
            colorAttachments: [{
                view: context.getCurrentTexture().createView(),
                clearValue: { r: 0.2, g: 0.2, b: 0.2, a: 1.0 },
                loadOp: 'clear',
                storeOp: 'store'
            }]
        });

        renderPass.setPipeline(pipeline);
        renderPass.setBindGroup(0, uniformBindGroup);
        renderPass.setVertexBuffer(0, particleBuffer);
        renderPass.setIndexBuffer(indexBuffer, 'uint32');
        renderPass.drawIndexed(clothData.indices.length);

        renderPass.end();

        device.queue.writeBuffer(simParamsBuffer, 0, new Float32Array([
            performance.now() / 1000,
            document.getElementById('gravityToggle').checked ? 1.0 : 0.0,
            clothRes,
            2.0 / clothRes,
            parseFloat(document.getElementById('freqSlider').value),
            parseFloat(document.getElementById('ampSlider').value)
        ]));
        device.queue.submit([commandEncoder.finish()]);
        
        requestAnimationFrame(render);
    }

    render();
}

init();
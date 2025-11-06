def fix_sdxl_compat():
    """Fix SDXL compatibility across diffusers versions"""
    try:
        from diffusers import StableDiffusionXLPipeline
        if not hasattr(StableDiffusionXLPipeline, 'do_classifier_free_guidance'):
            StableDiffusionXLPipeline.do_classifier_free_guidance = True
    except:
        pass

fix_sdxl_compat()  # Auto-run on import
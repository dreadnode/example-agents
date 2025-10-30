<?php
/**
 * Animations Settings Module
 * Settings > Widget Options :: Animation
 *
 * @copyright   Copyright (c) 2016, Jeffrey Carandang
 * @since       3.0
*/

// Exit if accessed directly
if ( ! defined( 'ABSPATH' ) ) exit;

/**
 * Create Card Module for Animation Options
 *
 * @since 3.0
 * @return void
 */
if( !function_exists( 'widgetopts_settings_animation' ) ):
	function widgetopts_settings_animation(){ ?>
		<li class="widgetopts-module-card widgetopts-module-type-pro" data-module-id="animation">
			<div class="widgetopts-module-card-content">
				<a href="<?php echo apply_filters('widget_options_site_url', trailingslashit(WIDGETOPTS_PLUGIN_WEBSITE).'features/wordpress-widget-animations/');?>" target="_blank" class="widgetopts-pro-upsell"></a>
				<h2><?php _e( 'Animation', 'widget-options' );?></h2>
				<div class="widgetopts-pro-label"><span class="dashicons dashicons-lock"></span></div>
				<p class="widgetopts-module-desc">
					<?php _e( 'Add CSS animation effect to your widgets on page load or page scroll.', 'widget-options' );?>
				</p>
			</div>
		</li>
	    <?php
	}
	add_action( 'widgetopts_module_cards', 'widgetopts_settings_animation', 130 );
endif;
?>
